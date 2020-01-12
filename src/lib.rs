#[macro_use]
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
mod zipper;
mod cata;
mod identity;
mod monodepth;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    cata::register::<identity::Identity>(plugin)?;
    cata::register::<monodepth::MonoDepth>(plugin)?;
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
