use std::{
    cell::RefCell,
    marker::PhantomData,
    sync::{atomic::AtomicPtr, mpsc, Arc},
};

use super::FrameInfo;
use anyhow::{anyhow, Context};
use log::warn;
use seize::{Collector, Guard};
use vulkano::sync::GpuFuture;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Control {
    /// Pause the camera thread so it's not capturing new frames, but keep the camera device open
    Pause,
    /// Resume capturing
    Resume,
}

pub struct CameraThread<PP: crate::pipeline::PostprocessPipeline> {
    control: Option<mpsc::Sender<Control>>,
    frame: Arc<AtomicPtr<FrameInfo<PP::Image>>>,
    collector: Arc<Collector>,
    _postprocessor: PhantomData<PP>,
}

thread_local! {
    static FREE_FRAME_LIST: RefCell<(Vec<*mut ()>, bool)> = const { RefCell::new((Vec::new(), false)) };
}

impl<PP: crate::pipeline::PostprocessPipeline + Send + 'static> CameraThread<PP> {
    pub fn new(camera: v4l::Device, splash: PP::Image, postprocessor: PP) -> Self {
        let (tx, rx) = mpsc::channel();
        let frame = Box::new(FrameInfo {
            frame: splash,
            frame_time: std::time::Instant::now(),
        });
        let frame = Arc::new(AtomicPtr::new(Box::into_raw(frame)));
        let collector = Arc::new(Collector::new());
        std::thread::spawn({
            let frame = frame.clone();
            let collector = collector.clone();
            move || Self::run(frame, rx, camera, postprocessor, collector)
        });

        Self {
            frame,
            control: Some(tx),
            collector,
            _postprocessor: PhantomData,
        }
    }
    pub fn with_frame<T: 'static>(&self, f: impl FnOnce(&FrameInfo<PP::Image>) -> T) -> T {
        let guard = self.collector.enter();
        let frame = guard.protect(&self.frame, std::sync::atomic::Ordering::Acquire);
        // SAFETY: protected by seize guard.
        f(unsafe { &*frame })
    }
    pub fn pause(&self) -> anyhow::Result<()> {
        let Some(control) = &self.control else {
            return Err(anyhow!("Camera thread already stopped"));
        };
        Ok(control.send(Control::Pause)?)
    }
    pub fn resume(&self) -> anyhow::Result<()> {
        let Some(control) = &self.control else {
            return Err(anyhow!("Camera thread already stopped"));
        };
        Ok(control.send(Control::Resume)?)
    }
    pub fn exit(&mut self) {
        self.control.take();
    }
    fn run(
        frame: Arc<AtomicPtr<FrameInfo<PP::Image>>>,
        control: mpsc::Receiver<Control>,
        camera: v4l::Device,
        postprocessor: PP,
        collector: Arc<Collector>,
    ) {
        let Err(e) = Self::run_inner(frame, control, camera, postprocessor, collector) else {
            return;
        };
        warn!("Camera thread stopped: {e}");
    }
    fn run_inner(
        frame: Arc<AtomicPtr<FrameInfo<PP::Image>>>,
        control: mpsc::Receiver<Control>,
        camera: v4l::Device,
        postprocessor: PP,
        collector: Arc<Collector>,
    ) -> anyhow::Result<()> {
        let mut first_frame_time = None;
        // We want to make the latency as low as possible, so only set a single buffer.
        let mut video_stream =
            v4l::prelude::MmapStream::with_buffers(&camera, v4l::buffer::Type::VideoCapture, 1)
                .context("cannot open camera mmap stream")?;
        FREE_FRAME_LIST.with_borrow_mut(|(_, started)| *started = true);
        Ok(loop {
            if let Some(c) = match control.try_recv() {
                Ok(c) => Some(c),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => break,
            } {
                match c {
                    Control::Pause => {
                        let Ok(c) = control.recv() else {
                            break;
                        };
                        assert_eq!(c, Control::Resume, "unexpected command {c:?}");
                    }
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
            let reuse_buffer = FREE_FRAME_LIST.with_borrow_mut(|(b, _)| b.pop());
            let new_frame = if let Some(frame_ptr) = reuse_buffer {
                // SAFETY: FREE_FRAME_LIST is a thread local, and from the camera thread, we
                // only ever put *mut FrameInfo<PP::Image> (erased as *mut ()) into it.
                let frame_ptr = frame_ptr.cast::<FrameInfo<PP::Image>>();
                let mut frame = unsafe { Box::from_raw(frame_ptr) };
                (*frame).frame_time = frame_time;
                frame
            } else {
                Box::new(FrameInfo {
                    frame: postprocessor.allocate_image()?,
                    frame_time: std::time::Instant::now(),
                })
            };
            postprocessor
                .postprocess(&frame_data, &new_frame.frame)?
                .then_signal_fence()
                .wait(None)?;
            let new_frame = Box::into_raw(new_frame);
            let guard = collector.enter();
            let old_frame = guard.swap(&frame, new_frame, std::sync::atomic::Ordering::Release);
            // SAFETY: old_frame will not be available to new readers.
            unsafe {
                collector.retire(old_frame, |value, _| {
                    FREE_FRAME_LIST.with_borrow_mut(|(list, started)| {
                        if !*started {
                            warn!("reclaimer called not from the camera thread");
                            Box::from_raw(value);
                            return;
                        }
                        list.push(value.cast());
                    })
                })
            };
            drop(guard);
            // log::debug!("got camera frame {}", frame_data.len());
        })
    }
}
