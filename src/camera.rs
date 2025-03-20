use std::{
    any,
    cell::RefCell,
    marker::PhantomData,
    sync::{atomic::AtomicPtr, mpsc, Arc, OnceLock},
    thread::JoinHandle,
};

use crate::pipeline::{self, PostprocessPipeline};

use super::FrameInfo;
use anyhow::{anyhow, bail, Context};
use arc_swap::{ArcSwap, ArcSwapOption, Guard};
use log::warn;
use smallvec::SmallVec;
use vulkano::{image::Image, sync::GpuFuture};

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Control {
    /// Pause the camera thread so it's not capturing new frames, but keep the camera device open
    Pause,
    /// Resume capturing
    Resume,
    Exit,
}

pub struct CameraThread {
    control: mpsc::Sender<Control>,
    frame: ArcSwap<FrameInfo>,
}

static CAMERA_THREAD: OnceLock<CameraThread> = OnceLock::new();

pub fn start(
    camera: v4l::Device,
    splash: Arc<Image>,
    postprocessor: Box<dyn PostprocessPipeline + Send>,
) -> anyhow::Result<(&'static CameraThread, JoinHandle<()>)> {
    let mut initialized = None;
    let ret = CAMERA_THREAD.get_or_init(|| {
        let (camera, thread) = CameraThread::new(camera, splash, postprocessor);
        initialized = Some(thread);
        camera
    });
    let Some(join) = initialized else {
        bail!("Camera thread already started!");
    };
    Ok((ret, join))
}

impl CameraThread {
    fn new(
        camera: v4l::Device,
        splash: Arc<Image>,
        postprocessor: Box<dyn PostprocessPipeline + Send>,
    ) -> (Self, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel();
        let frame = FrameInfo {
            frame: splash,
            frame_time: std::time::Instant::now(),
        };
        let frame = ArcSwap::new(Arc::new(frame));
        let thread = std::thread::spawn(move || Self::run(rx, camera, postprocessor));

        (Self { frame, control: tx }, thread)
    }
    pub fn frame(&self) -> Guard<Arc<FrameInfo>> {
        self.frame.load()
    }
    pub fn pause(&self) -> anyhow::Result<()> {
        Ok(self.control.send(Control::Pause)?)
    }
    pub fn resume(&self) -> anyhow::Result<()> {
        Ok(self.control.send(Control::Resume)?)
    }
    pub fn exit(&self) -> anyhow::Result<()> {
        Ok(self.control.send(Control::Exit)?)
    }
    fn run(
        control: mpsc::Receiver<Control>,
        camera: v4l::Device,
        postprocessor: Box<dyn PostprocessPipeline>,
    ) {
        let Err(e) = Self::run_inner(control, camera, postprocessor) else {
            return;
        };
        warn!("Camera thread stopped: {e:#}");
    }
    fn run_inner(
        control: mpsc::Receiver<Control>,
        camera: v4l::Device,
        postprocessor: Box<dyn PostprocessPipeline>,
    ) -> anyhow::Result<()> {
        let mut render_doc = renderdoc::RenderDoc::<renderdoc::V100>::new().ok();
        if render_doc.is_some() {
            log::info!("RenderDoc loaded");
        }
        let mut first_frame_time = None;
        // We want to make the latency as low as possible, so only set a single buffer.
        let mut video_stream =
            v4l::prelude::MmapStream::with_buffers(&camera, v4l::buffer::Type::VideoCapture, 1)
                .context("cannot open camera mmap stream")?;
        let mut is_splash = true;
        {
            let c = control.recv()?;
            assert_eq!(c, Control::Resume, "unexpected command {c:?}");
        }
        const MAX_POOL_SIZE: usize = 2;
        let mut pool = SmallVec::<[Arc<FrameInfo>; 2]>::new();

        let find_free = |pool: &mut SmallVec<_>| {
            // Find unused image from pool
            let index = pool.iter_mut().position(|i| Arc::get_mut(i).is_some())?;
            Some(pool.swap_remove(index))
        };
        loop {
            if let Some(c) = match control.try_recv() {
                Ok(c) => Some(c),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => break Ok(()),
            } {
                match c {
                    Control::Pause => {
                        let Ok(c) = control.recv() else {
                            break Ok(());
                        };
                        assert_eq!(c, Control::Resume, "unexpected command {c:?}");
                    }
                    Control::Exit => break Ok(()),
                    Control::Resume => panic!("unexpected resume"),
                }
            }
            log::trace!("getting camera frame");
            let (frame_data, metadata) = v4l::io::traits::CaptureStream::next(&mut video_stream)?;
            let frame_time = if let Some((camera_reference, reference)) = first_frame_time {
                let camera_elapsed =
                    std::time::Duration::from(metadata.timestamp) - camera_reference;
                reference + camera_elapsed
            } else {
                let now = std::time::Instant::now();
                first_frame_time = Some((metadata.timestamp.into(), now));
                now
            };
            log::trace!("got camera frame {:?}", frame_time);
            let mut new_frame = find_free(&mut pool)
                .map(Ok::<_, anyhow::Error>)
                .unwrap_or_else(|| {
                    log::info!("Allocated new frame");
                    Ok(FrameInfo {
                        frame: postprocessor.allocate_image()?,
                        frame_time: std::time::Instant::now(),
                    }
                    .into())
                })?;
            {
                // This `get_mut` can't fail, `find_free` returns `Arc` that's unique, if it
                // returns `None` then `new_frame` will be a newly allocated `Arc`.
                let new_frame = Arc::get_mut(&mut new_frame).unwrap();
                if let Some(rd) = render_doc.as_mut() {
                    rd.trigger_capture();
                }
                new_frame.frame_time = frame_time;
                postprocessor
                    .postprocess(frame_data, new_frame.frame.clone())?
                    .then_signal_fence()
                    .wait(None)?;
            }
            let old_frame = CAMERA_THREAD.get().unwrap().frame.swap(new_frame);
            // splash image isn't allocated from the pipeline, so it can't be reused.
            if !is_splash && pool.len() < MAX_POOL_SIZE {
                pool.push(old_frame);
            } else {
                log::info!("Releasing frame image, is_splash {is_splash}");
            }
            is_splash = false;
            // log::debug!("got camera frame {}", frame_data.len());
        }
    }
}
