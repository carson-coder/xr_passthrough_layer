use std::{
    sync::{mpsc, Arc},
    thread::JoinHandle,
};

use super::FrameInfo;
use anyhow::{anyhow, Context};
use arc_swap::{ArcSwap, Guard};
use glam::UVec2;
use log::{info as debug, warn};
use smallvec::SmallVec;
use v4l::video::Capture;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Control {
    /// Pause the camera thread so it's not capturing new frames, but keep the camera device open
    Pause,
    /// Resume capturing
    Resume,
    Exit,
}

pub struct CameraThread {
    frame: Arc<ArcSwap<FrameInfo>>,
    control: mpsc::Sender<Control>,
    join: JoinHandle<()>,
}

impl CameraThread {
    pub fn new(camera: v4l::Device, splash_png: &[u8]) -> Self {
        let (tx, rx) = mpsc::channel();
        let img = image::load_from_memory_with_format(splash_png, image::ImageFormat::Png)
            .unwrap()
            .into_rgba8();
        let extent = [img.width(), img.height()];
        assert!(extent[0] % 2 == 0);

        let frame = FrameInfo {
            frame: img.into_raw(),
            frame_time: std::time::Instant::now(),
            needs_postprocess: false, // splash image doesn't need postprocessing
            size: UVec2::new(extent[0] / 2, extent[1]),
        };
        let frame = Arc::new(ArcSwap::new(Arc::new(frame)));
        let join = std::thread::spawn({
            let frame = frame.clone();
            move || Self::run(rx, camera, frame)
        });
        Self {
            frame,
            join,
            control: tx,
        }
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
    pub fn exit(self) -> anyhow::Result<()> {
        self.control.send(Control::Exit)?;
        self.join.join().map_err(|e| anyhow!("{e:?}"))
    }

    fn run(control: mpsc::Receiver<Control>, camera: v4l::Device, frame: Arc<ArcSwap<FrameInfo>>) {
        // hold the thread until we get a go signal
        match control.recv() {
            Err(std::sync::mpsc::RecvError) | Ok(Control::Exit) => {
                debug!("camera thread stopped early");
                return;
            }
            Ok(Control::Resume) => (),
            Ok(Control::Pause) => panic!("Invalid pause command"),
        }
        let Err(e) = Self::run_inner(control, camera, frame) else {
            debug!("camera thread stopped");
            return;
        };
        warn!("Camera thread stopped: {e:#}");
    }
    fn run_inner(
        control: mpsc::Receiver<Control>,
        camera: v4l::Device,
        frame: Arc<ArcSwap<FrameInfo>>,
    ) -> anyhow::Result<()> {
        let mut first_frame_time = None;
        // We want to make the latency as low as possible, so only set a single buffer.
        let mut video_stream =
            v4l::prelude::MmapStream::with_buffers(&camera, v4l::buffer::Type::VideoCapture, 1)
                .context("cannot open camera mmap stream")?;
        let mut is_splash = true;
        const MAX_POOL_SIZE: usize = 2;
        let mut pool = SmallVec::<[Arc<FrameInfo>; 2]>::new();
        let camera_format = camera.format()?;
        if camera_format.width % 2 != 0 {
            return Err(anyhow!("Camera width is not even"));
        }

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
            let new_frame = find_free(&mut pool)
                .map(|mut fi| {
                    log::trace!("Reusing frame");
                    let mfi = Arc::get_mut(&mut fi).unwrap();
                    mfi.frame.copy_from_slice(frame_data);
                    mfi.frame_time = frame_time;
                    mfi.needs_postprocess = true;
                    fi
                })
                .unwrap_or_else(|| {
                    log::debug!("Allocated new frame");
                    FrameInfo {
                        frame: frame_data.to_vec(),
                        frame_time: std::time::Instant::now(),
                        needs_postprocess: true,
                        size: UVec2::new(camera_format.width / 2, camera_format.height),
                    }
                    .into()
                });
            let old_frame = frame.swap(new_frame);
            // splash image isn't necessarily `frame_data` sized, so don't reuse it.
            // if we already have too many frames, just release the old one.
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
