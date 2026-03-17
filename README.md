# xr_passthrough_layer

A OpenXR API layer adding camera passthrough support.

> [!WARNING]
> This is early development quality software.

## Dependencies

- libudev
- shaderc
- rustc & cargo

## Build & Installation

Run:

```bash
cargo xtask install
```

This will install the API layer shared library to `~/.local/lib/xr_passthrough_layer/`, and the API layer json file to `~/.local/share/openxr/1/api_layers`. The API layer will be called `XR_APILAYER_YX_passthrough`

## Usage

Load this layer when creating an OpenXR instance. Or use the `XR_ENABLE_API_LAYERS` environment variable.

## Project status

What hardware is supported:

- The Valve Index HMD. Other HMD cameras may work but nothing else has been tested.

API layer features:

- The `ALPHA_BLEND` OpenXR environment blend mode. Loading this API layer adds `ALPHA_BLEND` into the supported environment blend mode. Applications will be able to make use of this mode to show camera passthrough in the background.

Future plan:

- Passthrough extension support. e.g. `XR_FB_passthrough` and/or `XR_HTC_passthrough`.
- Optimization. e.g. use dmabuf for camera capture.
