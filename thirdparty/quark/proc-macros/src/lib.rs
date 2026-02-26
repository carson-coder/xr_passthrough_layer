use heck::ToSnakeCase;
use quote::{format_ident, quote};
use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use proc_macro2::{token_stream, Delimiter, Group, Ident, Punct, TokenStream, TokenTree};

fn parse_ident_map(tokens: TokenStream, map: &mut HashMap<String, String>) {
    let mut tokens = tokens.into_iter();
    while let Ok(k) = tokens.expect_ident() {
        tokens.expect_punct(':').unwrap();
        let v = tokens.expect_ident().unwrap();
        let old = map.insert(k.to_string(), v.to_string());
        if let Some(old) = old {
            panic!("Duplicate entry for key {old}");
        }
        let Ok(_) = tokens.expect_punct(',') else {
            break;
        };
    }
}

fn delimiter_start(d: Delimiter) -> char {
    match d {
        Delimiter::Parenthesis => '(',
        Delimiter::Bracket => '[',
        Delimiter::Brace => '{',
        Delimiter::None => '\0',
    }
}

trait TokenStreamExt: Iterator<Item = TokenTree> {
    fn expect_punct(&mut self, p: char) -> Result<Punct, &'static str> {
        let next = self.next().ok_or("Unexpected end of tokens")?;
        match next {
            TokenTree::Punct(next) if next.as_char() == p => Ok(next),
            other => panic!("Unexpected token {other}, expecting {p}"),
        }
    }
    fn expect_group(&mut self, delimiter: Delimiter) -> Result<Group, &'static str> {
        let next = self.next().ok_or("Unexpected end of tokens")?;
        match next {
            TokenTree::Group(next) if next.delimiter() == delimiter => Ok(next),
            TokenTree::Group(next) => {
                panic!("Unexpected token {}", delimiter_start(next.delimiter()))
            }
            other => panic!(
                "Unexpected token {other}, expecting {}",
                delimiter_start(delimiter)
            ),
        }
    }
    fn expect_ident(&mut self) -> Result<Ident, &'static str> {
        let next = self.next().ok_or("Unexpected end of tokens")?;
        match next {
            TokenTree::Ident(next) => Ok(next),
            other => panic!("Unexpected token {other}, expecting ident"),
        }
    }
}

impl TokenStreamExt for token_stream::IntoIter {}

const OPENXR_SPEC: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/xr.xml"));

mod spec {
    #![allow(dead_code)]
    use serde::Deserialize;
    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum SignatureParts {
        Type(String),
        Name(String),
        Enum(String),
        #[serde(rename = "$text")]
        Other(String),
    }

    #[derive(Deserialize, Debug)]
    pub(super) struct Signature {
        #[serde(rename = "$value")]
        pub(super) parts: Vec<SignatureParts>,
    }
    #[derive(Deserialize, Debug)]
    pub(super) struct Command {
        pub(super) proto: Option<Signature>,
        pub(super) param: Option<Vec<Signature>>,
        #[serde(rename = "@name")]
        pub(super) name: Option<String>,
        #[serde(rename = "@alias")]
        pub(super) alias: Option<String>,
    }
    impl Command {
        pub(super) fn name(&self) -> &str {
            if let Some(name) = &self.name {
                return name;
            }

            let proto = self.proto.as_ref().expect("Malformed openxr spec");
            proto
                .parts
                .iter()
                .find_map(|p| match p {
                    SignatureParts::Name(name) => Some(name),
                    _ => None,
                })
                .expect("Malformed openxr spec")
        }
    }

    #[derive(Deserialize, Debug)]
    pub(super) struct ExtensionCommand {
        #[serde(rename = "@name")]
        pub(super) name: String,
    }

    #[derive(Deserialize, Debug)]
    pub(super) struct ExtensionRequire {
        #[serde(default)]
        pub(super) command: Vec<ExtensionCommand>,
    }

    #[derive(Deserialize, Debug)]
    pub(super) struct Extension {
        #[serde(rename = "@name")]
        pub(super) name: String,
        pub(super) require: Vec<ExtensionRequire>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum TypePart {
        Name(String),
        #[serde(rename = "$text", other)]
        Other,
    }

    #[derive(Deserialize, Debug)]
    pub(super) struct Type {
        #[serde(rename = "@category")]
        pub(super) category: Option<String>,
        #[serde(rename = "@parent")]
        pub(super) parent: Option<String>,

        #[serde(rename = "$value", default)]
        pub(super) parts: Vec<TypePart>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum Section {
        Commands {
            command: Vec<Command>,
        },
        Enums,
        Comment,
        Vendorids,
        Tags,
        Types {
            #[serde(rename = "$value")]
            types: Vec<Type>,
        },
        InteractionProfiles,
        Feature,
        Extensions {
            extension: Vec<Extension>,
        },

        #[serde(rename = "$text")]
        Other(String),
    }
    #[derive(Deserialize, Debug)]
    pub(super) struct Spec {
        #[serde(rename = "$value")]
        pub(super) sections: Vec<Section>,
    }
}

#[derive(Debug)]
enum Type {
    Const(Box<Type>),
    Ptr(Box<Type>),
    Base(String),
}

impl Type {
    fn base(&self) -> &str {
        match self {
            Self::Const(inner) => inner.base(),
            Self::Ptr(inner) => inner.base(),
            Self::Base(base) => base,
        }
    }
    fn as_const(&self) -> Option<&Self> {
        match self {
            Self::Const(inner) => Some(inner),
            _ => None,
        }
    }
    fn as_ptr(&self) -> Option<&Self> {
        match self {
            Self::Ptr(inner) => Some(inner),
            _ => None,
        }
    }
    fn as_base(&self) -> Option<&str> {
        match self {
            Self::Base(base) => Some(base),
            _ => None,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const(inner) => match &**inner {
                Self::Ptr(base) => write!(f, "*const {base}"),
                _ => panic!("const not attached to pointer"),
            },
            Self::Ptr(inner) => write!(f, "*mut {inner}"),
            Self::Base(base) => write!(f, "{base}"),
        }
    }
}

impl From<&'_ [spec::SignatureParts]> for Type {
    fn from(value: &[spec::SignatureParts]) -> Self {
        match &value[0] {
            spec::SignatureParts::Other(q) if q == "const" => {
                Self::Const(Box::new(value[1..].into()))
            }
            spec::SignatureParts::Type(t) => {
                if let spec::SignatureParts::Other(p) = &value[1] {
                    assert_eq!(p, "*", "Malformed openxr spec");
                    Self::Ptr(Box::new(Self::Base(t.clone())))
                } else if value.len() > 2 {
                    if let spec::SignatureParts::Other(p) = &value[2] {
                        assert_eq!(
                            p, "[",
                            "Malformed openxr spec, unexpected \"{p}\" after name"
                        );
                        // Array type, treat as pointer
                        Self::Ptr(Box::new(Self::Base(t.clone())))
                    } else {
                        panic!("Unexpected {:?} after name", value[2]);
                    }
                } else {
                    Self::Base(t.clone())
                }
            }
            _ => panic!("Malformed openxr spec"),
        }
    }
}

#[derive(Debug)]
struct Signature {
    ty: Type,
    name: String,
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}

impl From<&'_ [spec::SignatureParts]> for Signature {
    fn from(value: &'_ [spec::SignatureParts]) -> Self {
        let ty: Type = value.into();
        let name = value
            .iter()
            .find_map(|v| {
                if let spec::SignatureParts::Name(name) = v {
                    Some(name)
                } else {
                    None
                }
            })
            .expect("Malformed openxr spec, can't find name in signature");
        Self {
            ty,
            name: name.to_string(),
        }
    }
}

use spec::Spec;

#[derive(Debug)]
struct Command {
    proto: Signature,
    params: Vec<Signature>,
}

#[derive(Default, Debug)]
struct CreateCommand {
    name: String,
    parent_ty: String,
    create_info_ty: String,
    out_handle_ty: String,
}

static CREATE_EXCLUDE: phf::Set<&'static str> = phf::phf_set! {
    // For some reason openxr-sys didn't have types for these functions...
    "xrCreateSceneMSFT",
    "xrCreateSceneObserverMSFT",
};

/// Filter out some `xrCreate` we don't know how to handle
fn check_signature(cmd: &Command, parent_of: &HashMap<String, String>) -> Option<CreateCommand> {
    if cmd.proto.name == "xrCreateApiLayerInstance" {
        // Special case, `xrCreateApiLayerInstance` doesn't take a parent handle, because
        // instance is the first handle created, and it takes 2 create infos.
        return Some(CreateCommand {
            name: cmd.proto.name.clone(),
            ..Default::default()
        });
    }
    if cmd.params.iter().any(|p| !p.ty.base().starts_with("Xr")) {
        return None;
    }
    if cmd.proto.ty.as_base()? != "XrResult" {
        return None;
    }

    // Create function should all look like this:

    // xrCreate<Handle>(parent: Xr<ParentHandle>, create_info: *const <Handle>CreateInfo, out_handle: *mut <Handle>)
    let [parent, create_info, out_handle]: &[Signature; 3] = cmd.params[..].try_into().ok()?;
    let parent_ty = parent.ty.as_base()?.to_string();
    if !parent_ty.starts_with("Xr") {
        return None;
    }
    let create_info_ty = create_info.ty.as_const()?.as_ptr()?.as_base()?.to_string();
    let out_handle_ty = out_handle.ty.as_ptr()?.as_base()?.to_string();
    if out_handle_ty == "XrAsyncRequestIdFB" {
        // Not the actual handle, but some sort of async request handle
        return None;
    }
    if parent_ty
        != *parent_of
            .get(&out_handle_ty)
            .expect("Missing parent type from spec")
    {
        // First parameter is not parent.
        return None;
    }

    Some(CreateCommand {
        name: cmd.proto.name.clone(),
        parent_ty,
        create_info_ty,
        out_handle_ty,
    })
}

struct ParsedSpec {
    create_commands: Vec<CreateCommand>,
    command_to_extension: HashMap<String, String>,
    /// Map handle type to its parent type
    parent_of: HashMap<String, String>,
    aliases: HashMap<String, String>,
    aliases_of: HashMap<String, HashSet<String>>,
}

static PARSED_SPEC: LazyLock<ParsedSpec> = LazyLock::new(|| {
    // We scan the spec for `xrCreate*` commands, and generate wrappers for them.
    let source = OPENXR_SPEC;
    let mut reader = std::io::Cursor::new(source);
    #[allow(const_item_mutation)]
    let spec: Spec = quick_xml::de::from_reader(&mut reader).unwrap_or_else(|e| {
        panic!(
            "Failed to parse openxr spec: {e:#}, {:?}",
            std::str::from_utf8(
                &source[(reader.position() - 100) as usize..(reader.position() + 100) as usize]
            )
        )
    });
    // Count if there is more than 1 way to create a given handle, if so, we can't handle this
    // case.
    let command_to_extension = spec
        .sections
        .iter()
        .filter_map(|s| {
            if let spec::Section::Extensions { extension } = s {
                Some(extension)
            } else {
                None
            }
        })
        .flat_map(|es| {
            es.iter().flat_map(|e| {
                e.require.iter().flat_map(|r| {
                    r.command
                        .iter()
                        .map(|c| (c.name.to_string(), e.name.to_string()))
                })
            })
        })
        .collect();
    let parent_of = spec
        .sections
        .iter()
        .filter_map(|s| {
            if let spec::Section::Types { types } = s {
                Some(types)
            } else {
                None
            }
        })
        .flat_map(|t| t.iter())
        .map(|t| {
            let name = t.parts.iter().find_map(|p| {
                if let spec::TypePart::Name(name) = p {
                    Some(name.as_str())
                } else {
                    None
                }
            });
            (name, t.category.as_deref(), &t.parent)
        })
        .filter(|(name, cat, _)| *name != Some("XrInstance") && *cat == Some("handle"))
        .map(|(name, _, parent)| {
            (
                name.expect("Handle type has no name").to_string(),
                parent.clone().expect("Handle type has no parent"),
            )
        })
        .collect::<HashMap<_, _>>();
    let create_commands = {
        let mut create_count = HashMap::<_, u32>::new();

        let mut cmds = spec
            .sections
            .iter()
            .filter_map(|s| match s {
                spec::Section::Commands { command } => Some(command),
                _ => None,
            })
            .flat_map(|c| c.iter())
            .filter(|c| c.name().starts_with("xrCreate") && !CREATE_EXCLUDE.contains(c.name()))
            .filter_map(|cmd| {
                if cmd.alias.is_some() {
                    return None;
                };
                let proto: Signature =
                    cmd.proto.as_ref().expect("No alias nor proto").parts[..].into();
                let params = cmd
                    .param
                    .as_ref()
                    .expect("No params")
                    .iter()
                    .map(|p| p.parts[..].into())
                    .collect();
                Some(Command { proto, params })
            })
            .filter_map(|cmd| check_signature(&cmd, &parent_of))
            .collect::<Vec<_>>();

        for cmd in &cmds {
            *create_count.entry(cmd.out_handle_ty.clone()).or_default() += 1;
        }
        cmds.retain(|cmd| *create_count.get(&cmd.out_handle_ty).unwrap() == 1);
        cmds
    };

    let aliases = spec
        .sections
        .iter()
        .filter_map(|s| {
            if let spec::Section::Commands { command } = s {
                Some(command)
            } else {
                None
            }
        })
        .flat_map(|c| c.iter())
        .filter_map(|cmd| {
            cmd.alias
                .as_ref()
                .map(|alias| (cmd.name().to_string(), alias.to_string()))
        })
        .collect::<HashMap<_, _>>();
    let mut aliases_of = HashMap::<String, HashSet<String>>::new();
    for (cmd, alias) in &aliases {
        let aliases = aliases_of.entry(alias.clone()).or_default();
        aliases.insert(cmd.clone());
    }

    ParsedSpec {
        create_commands,
        command_to_extension,
        parent_of,
        aliases,
        aliases_of,
    }
});

#[allow(unused_macros)]
macro_rules! dpy {
    ($val:expr $(,)?) => {
        // Use of `match` here is intentional because it affects the lifetimes
        // of temporaries - https://stackoverflow.com/a/48732525/1063961
        match $val {
            tmp => {
                eprintln!(
                    "[{}:{}:{}] {} = {}",
                    file!(),
                    line!(),
                    column!(),
                    stringify!($val),
                    &tmp
                );
                tmp
            }
        }
    };
}

/// Generate wrappers for `xrCreate*` functions.
#[proc_macro]
pub fn gen_create_wrapper(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let crate_ = TokenStream::from(tokens)
        .into_iter()
        .next()
        .expect("missing $crate");
    PARSED_SPEC
        .create_commands
        .iter()
        .map(|cmd| {
            let our_name =
                format_ident!("{}", cmd.name.strip_prefix("xr").unwrap().to_snake_case());
            let (params, args, handle_ty, reg) = if our_name == "create_api_layer_instance" {
                (
                    quote! {
                        info: *const openxr::sys::InstanceCreateInfo,
                        api_layer_info: *const openxr::sys::loader::ApiLayerCreateInfo,
                        out_handle: *mut openxr::sys::Instance,
                    },
                    quote! {
                        info, api_layer_info
                    },
                    format_ident!("Instance"),
                    quote! {
                        let handle = high.as_handle();
                        #crate_::register_object(high, handle, hook)
                    },
                )
            } else {
                let out_handle = format_ident!("{}", cmd.out_handle_ty.strip_prefix("Xr").unwrap());
                (
                    {
                        let parent = format_ident!("{}", cmd.parent_ty.strip_prefix("Xr").unwrap());
                        let create_info =
                            format_ident!("{}", cmd.create_info_ty.strip_prefix("Xr").unwrap());
                        quote! {
                            parent: openxr::sys::#parent,
                            create_info: *const openxr::sys::#create_info,
                            out_handle: *mut openxr::sys::#out_handle,
                        }
                    },
                    quote! { parent, create_info },
                    out_handle,
                    quote! { #crate_::register_object(high, parent, hook) },
                )
            };
            quote! {
                pub unsafe extern "system" fn #our_name<P>(
                    #params
                ) -> openxr::sys::Result
                where
                    P: #crate_::Hook<Target = openxr::sys::#handle_ty>,
                {
                    let args = (#args, out_handle);
                    let (high, hook) = #crate_::try_xr!(
                        <<P as #crate_::Hook>::Factory as #crate_::Factory<P>>::create(args)
                    );
                    unsafe { #reg }
                    XrErr::SUCCESS
                }
            }
        })
        .collect::<proc_macro2::TokenStream>()
        .into()
}

/// Generate `impl Create` for openxr handle types.
#[proc_macro]
pub fn impl_create(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let crate_ = TokenStream::from(tokens)
        .into_iter()
        .next()
        .expect("missing $crate");
    PARSED_SPEC
        .create_commands
        .iter()
        .filter(|cmd| cmd.name != "xrCreateApiLayerInstance") // Create for Instance is written
        // manually
        .map(|cmd| {
            let handle_ty = format_ident!("{}", cmd.out_handle_ty.strip_prefix("Xr").unwrap());
            let parent = format_ident!("{}", cmd.parent_ty.strip_prefix("Xr").unwrap());
            let create_info = format_ident!("{}", cmd.create_info_ty.strip_prefix("Xr").unwrap());
            let params = quote! {
                openxr::sys::#parent,
                *const openxr::sys::#create_info,
                *mut openxr::sys::#handle_ty,
            };
            let factory_name = format_ident!("{}Factory", handle_ty);
            let fp = if let Some(extension) = PARSED_SPEC.command_to_extension.get(&cmd.name) {
                let ext = extension.strip_prefix("XR_").unwrap();
                let (tag, _) = ext.split_once("_").unwrap();
                let ext = format_ident!("{}", ext.to_lowercase());
                let our_name = format_ident!(
                    "{}",
                    cmd.name
                        .strip_prefix("xr")
                        .unwrap()
                        .strip_suffix(tag)
                        .unwrap()
                        .to_snake_case()
                );
                quote! {
                    i.exts().#ext.ok_or(#crate_::XrErr::ERROR_EXTENSION_NOT_PRESENT)?.#our_name
                }
            } else {
                let our_name =
                    format_ident!("{}", cmd.name.strip_prefix("xr").unwrap().to_snake_case());
                quote! {
                    i.fp().#our_name
                }
            };
            quote! {
                impl #crate_::sealed::Create for openxr::sys::#handle_ty {
                    type Args = (#params);
                }
                pub struct #factory_name;
                impl #crate_::sealed::HasFactory for openxr::sys::#handle_ty {
                    type Factory = #factory_name;
                }
                unsafe impl<Hook: #crate_::Hook<Target = openxr::sys::#handle_ty>> #crate_::Factory<Hook>
                    for #factory_name
                {
                    unsafe fn create(
                        args: <Hook::Target as #crate_::sealed::Create>::Args
                    ) -> Result<(<Hook::Target as #crate_::Low>::High, Hook), #crate_::XrErr> {
                        let i = #crate_::find_instance(args.0)?
                            .registered()?;
                        let r = (#fp)(args.0, args.1, args.2);
                        if r != #crate_::XrErr::SUCCESS {
                            return Err(r);
                        }
                        invoke_on_create(*args.2, args)
                    }
                }
            }
        })
        .collect::<TokenStream>()
        .into()
}

#[proc_macro]
pub fn gen_facades(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut tokens = TokenStream::from(tokens).into_iter();
    let crate_ = tokens.next().expect("missing $crate");
    let manual_impl = tokens
        .map(|t| {
            let TokenTree::Ident(ident) = t else {
                panic!("Unexpected token {t}")
            };
            ident.to_string()
        })
        .collect::<HashSet<_>>();

    PARSED_SPEC
        .create_commands
        .iter()
        .map(|cmd| {
            if cmd.name == "xrCreateApiLayerInstance" {
                return quote! {};
            }
            let handle_ty = cmd.out_handle_ty.strip_prefix("Xr").unwrap();
            if manual_impl.contains(handle_ty) {
                return quote! {};
            }
            let handle_ty = format_ident!("{handle_ty}");
            let create_info = format_ident!("{}", cmd.create_info_ty.strip_prefix("Xr").unwrap());
            quote! {
                unsafe impl Low for openxr::sys::#handle_ty {
                    type High = #crate_::Facade<openxr::sys::#handle_ty>;
                    type HighCreateInfo = openxr::sys::#create_info;
                    unsafe fn into_high(
                        self,
                        args: <Self as #crate_::sealed::Create>::Args,
                    ) -> Result<(Self::High, Self::HighCreateInfo), openxr::sys::Result> {
                        Ok((#crate_::Facade(self), *args.1))
                    }
                }
            }
        })
        .collect::<TokenStream>()
        .into()
}

/// For any given type we want to hook, its parent must also be hooked so we can manage its
/// lifetime correctly. This function expands `hooks` to make sure all parents are included.
fn find_closure(hooks: &mut HashMap<String, String>) -> Vec<String> {
    let mut ret = Vec::new();
    let mut before = 0;
    for k in hooks.keys() {
        if k == "Instance" {
            continue;
        }
        let name = format!("Xr{k}");
        let parent = PARSED_SPEC
            .parent_of
            .get(&name)
            .unwrap_or_else(|| panic!("Can't find parent of {name}"));
        let parent = parent.strip_prefix("Xr").unwrap();
        if !hooks.contains_key(parent) {
            ret.push(parent.to_string());
        }
    }
    let mut tmp = Vec::new();
    loop {
        for new_hook in &ret[before..] {
            hooks.insert(new_hook.to_string(), format!("Default{new_hook}Hook"));
            if new_hook == "Instance" {
                continue;
            }
            let name = format!("Xr{new_hook}");
            let parent = PARSED_SPEC.parent_of.get(&name).unwrap();
            if !hooks.contains_key(parent) {
                tmp.push(parent.strip_prefix("Xr").unwrap().to_string());
            }
        }
        if tmp.is_empty() {
            break;
        }
        before = ret.len();
        ret.append(&mut tmp);
    }
    ret
}

#[proc_macro]
pub fn gen_override_table(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut tokens = TokenStream::from(tokens).into_iter();
    let mut hooks = HashMap::new();
    let mut override_fns = HashMap::new();
    let crate_ = tokens.expect_ident().unwrap();
    let all_generated_creates = PARSED_SPEC
        .create_commands
        .iter()
        .map(|c| &c.name)
        .collect::<HashSet<_>>();
    tokens.expect_punct(';').unwrap();
    while let Ok(ident) = tokens.expect_ident() {
        tokens.expect_punct(':').unwrap();
        let group = tokens.expect_group(Delimiter::Brace).unwrap();

        match ident.to_string().as_str() {
            "hooks" => parse_ident_map(group.stream(), &mut hooks),
            "override_fns" => parse_ident_map(group.stream(), &mut override_fns),
            other => panic!("Unexpected field {other}"),
        }
        let Ok(_) = tokens.expect_punct(',') else {
            break;
        };
    }

    // We must hook Instance so we can implment create_api_layer_instance.
    let instance_data = format_ident!(
        "{}",
        hooks
            .entry("Instance".to_string())
            .or_insert_with(|| "DefaultInstanceHook".to_string())
    );

    let extra_hooks = find_closure(&mut hooks);

    let user_overrides: TokenStream = override_fns
        .into_iter()
        .map(|(k, v)| {
            let from = k.to_string();
            assert!(
                !all_generated_creates.contains(&from),
                "{from} is an auto generated function, don't override it, hook the handle instead."
            );
            if let Some(aliased) = PARSED_SPEC.aliases.get(&from) {
                panic!("{from} is an alias, don't override it, hooked aliased function {aliased} instead.");
            }
            if let Some(h) = from.strip_prefix("xrDestroy") {
                let create_name = format!("xrCreate{h}");
                assert!(
                    !all_generated_creates.contains(&create_name),
                    "{from} has auto generated destroy function, don't override it, hook the handle instead."
                );
            }
            let v = format_ident!("{v}");
            let aliases: TokenStream = PARSED_SPEC
                .aliases_of
                .get(&from)
                .iter()
                .flat_map(|a| a.iter())
                .map(|alias| {
                    quote! {
                        #alias => unsafe { std::mem::transmute(#v as *const ()) },
                    }
                }).collect();
            quote! {
                #from => unsafe { std::mem::transmute(#v as *const ()) },
                #aliases
            }
        })
        .collect();
    let creates: TokenStream = hooks
        .iter()
        .map(|(k, v)| {
            if k != "Instance" {
                let their_f = format!("xrCreate{k}");
                let our_f = format_ident!("create_{}", k.to_string().to_snake_case());
                let v = format_ident!("{v}");
                quote! {
                    #their_f => unsafe { std::mem::transmute(#crate_::wrappers::#our_f::<#v> as *const ()) },
                }
            } else {
                TokenStream::new()
            }
        })
        .collect();
    let destroys: TokenStream = hooks.into_keys().map(|k| {
        let their_f = format!("xrDestroy{k}");
        let k = format_ident!("{k}");
        quote! {
            #their_f => unsafe { std::mem::transmute(#crate_::wrappers::destroy_object::<openxr::sys::#k> as *const ()) },
        }
    }).collect();

    let default_hooks = extra_hooks
        .into_iter()
        .map(|t| {
            let name = format_ident!("Default{t}Hook");
            let handle = format_ident!("{t}");
            quote! {
                struct #name;
                impl #crate_::Hook for #name {
                    type Target = openxr::sys::#handle;
                    type Factory = #crate_::FactoryOf<Self>;
                    fn on_create(
                        _a: &<openxr::sys::#handle as #crate_::Low>::High,
                        _b: <openxr::sys::#handle as #crate_::Low>::HighCreateInfo,
                    ) -> Result<Self, openxr::sys::Result> {
                        Ok(Self)
                    }
                }
                unsafe impl #crate_::Hooked<#name> for openxr::sys::#handle {}
            }
        })
        .collect::<TokenStream>();

    let ret = quote! {
        const _: fn() = || {
        #default_hooks
        static OVERRIDE_TABLE: phf::Map<&'static str, openxr::sys::pfn::VoidFunction> =
            phf::phf_map! {
                #user_overrides
                #creates
                #destroys
            };
        #[allow(non_local_definitions)]
        unsafe impl #crate_::ApiLayerOverride for #instance_data {
            fn overrides() -> &'static phf::Map<&'static str, openxr::sys::pfn::VoidFunction> {
                &OVERRIDE_TABLE
            }
        }
        #[allow(non_snake_case)]
        #[no_mangle]
        pub unsafe fn xrNegotiateLoaderApiLayerInterface(
            loader_info: *const openxr::sys::loader::XrNegotiateLoaderInfo,
            api_layer_name: *const u8,
            api_layer_request: *mut openxr::sys::loader::XrNegotiateApiLayerRequest,
        ) -> openxr::sys::Result {
            match #crate_::negotiate_loader_api_layer_interface_impl::<
                #instance_data
            >(
                &*loader_info, api_layer_name, &mut *api_layer_request
            ) {
                Ok(()) => openxr::sys::Result::SUCCESS,
                Err(e) => e,
            }
        }};
    };
    ret.into()
}