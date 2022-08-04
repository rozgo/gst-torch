extern crate glib;
#[macro_use]
extern crate gstreamer as gst;
extern crate gstreamer_audio as gst_audio;
extern crate gstreamer_base as gst_base;
extern crate gstreamer_video as gst_video;

extern crate byte_slice_cast;
extern crate num_traits;

#[macro_use]
extern crate lazy_static;

#[macro_use]
mod registry;

extern crate rand;

mod caps;
mod cata;
mod facepose;
mod monodepth;
mod motiontransfer;
mod render;
mod salientobject;
mod semseg;
mod zipper;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    python_init();
    cata::register::<monodepth::MonoDepth>(plugin)?;
    cata::register::<semseg::SemSeg>(plugin)?;
    cata::register::<motiontransfer::MotionTransfer>(plugin)?;
    cata::register::<facepose::FacePose>(plugin)?;
    cata::register::<salientobject::SalientObject>(plugin)?;
    Ok(())
}

gst_plugin_define!(
    tensorflow,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);

fn python_init() {
    extern "C" {
        fn dlopen(filename: *const libc::c_char, flags: libc::c_int) -> *mut libc::c_void;
    }

    const RTLD_GLOBAL: libc::c_int = 0x00100;
    const RTLD_LAZY: libc::c_int = 0x00001;

    // make sure this path is null-terminated
    const LIBPYTHON: &'static str = "/usr/lib/x86_64-linux-gnu/libpython3.8.so\0";
    unsafe {
        dlopen(LIBPYTHON.as_ptr() as *const i8, RTLD_GLOBAL | RTLD_LAZY);
    }
}
