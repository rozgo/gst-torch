use glib;
use glib::prelude::*;
use glib::subclass;
use glib::subclass::prelude::*;
use gst;
use gst::prelude::*;
use gst::subclass::prelude::*;

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::caps::CapsDef;
use crate::registry::Registry;
use crate::zipper::Zipper;

pub trait Process {
    fn process(
        &mut self,
        inbuf: &Vec<gst::Buffer>,
        outbuf: &mut Vec<gst::Buffer>,
    ) -> Result<(), std::io::Error>;

    fn set_property(&mut self, property: &subclass::Property, value: &glib::Value);
}

struct State<T>
where
    T: 'static + Send + Default + Process + CapsDef,
{
    processor: T,
}

impl<T> Default for State<T>
where
    T: 'static + Send + Default + Process + CapsDef,
{
    fn default() -> State<T> {
        State::<T> {
            processor: Default::default(),
        }
    }
}

type PadMap = HashMap<gst::Pad, PadInfo>;

#[derive(Debug)]
struct PadInfo {
    name: &'static str,
    idx: usize,
    caps: gst::Caps,
}

pub struct Cata<T>
where
    T: 'static + Send + Default + Process + CapsDef,
{
    cat: gst::DebugCategory,
    src_pads: Mutex<PadMap>,
    sink_pads: Mutex<PadMap>,
    zipper: Mutex<Zipper>,
    state: Mutex<State<T>>,
}

impl<T> Cata<T>
where
    T: 'static + Send + Default + Process + CapsDef + Registry,
    Self: ElementImpl,
{
    fn cata_event(&self, pad: &gst::Pad, element: &gst::Element, event: gst::Event) -> bool {
        use gst::EventView;

        gst_log!(self.cat, obj: pad, "Handling event {:?}", event);

        let ret = match event.view() {
            EventView::FlushStart(..) => {
                let _ = self.stop(element);
                true
            }
            EventView::FlushStop(..) => {
                let (res, state, pending) = element.get_state(0.into());
                if res == Ok(gst::StateChangeSuccess::Success) && state == gst::State::Playing
                    || res == Ok(gst::StateChangeSuccess::Async) && pending == gst::State::Playing
                {
                    let _ = self.start(element);
                }
                true
            }
            EventView::Reconfigure(..) => true,
            EventView::Latency(..) => true,
            EventView::StreamStart(..) => true,
            EventView::Caps(..) => true,
            EventView::Tag(..) => true,
            EventView::Segment(..) => true,
            EventView::Qos(..) => true,
            EventView::StreamGroupDone(..) => true,
            EventView::Eos(..) => {
                let _ = self.stop(element);
                true
            }
            _ => false,
        };

        if ret {
            gst_log!(self.cat, obj: pad, "Handled event {:?}", event);
        } else {
            gst_log!(self.cat, obj: pad, "Didn't handle event {:?}", event);
        }

        ret
    }

    fn cata_query(
        &self,
        pad: &gst::Pad,
        _element: &gst::Element,
        query: &mut gst::QueryRef,
    ) -> bool {
        use gst::QueryView;

        gst_log!(self.cat, obj: pad, "Handling query {:?}", query);
        let ret = match query.view_mut() {
            QueryView::Latency(ref mut q) => {
                q.set(false, 1000000.into(), (100000000).into());
                true
            }
            QueryView::Scheduling(ref mut q) => {
                q.set(gst::SchedulingFlags::SEQUENTIAL, 1, -1, 0);
                q.add_scheduling_modes(&[gst::PadMode::Push]);
                true
            }
            QueryView::AcceptCaps(ref mut q) => {
                // TODO: validate what we are agreeing to
                q.set_result(true);
                true
            }
            QueryView::Caps(ref mut q) => {
                // Agree on caps
                let pads = match pad.get_direction() {
                    gst::PadDirection::Sink => self.sink_pads.lock().unwrap(),
                    gst::PadDirection::Src => self.src_pads.lock().unwrap(),
                    _ => panic!("Querying pad with unknown direction"),
                };
                let caps = &pads.get(pad).unwrap().caps;
                let caps = q
                    .get_filter()
                    .map(|f| f.intersect_with_mode(caps, gst::CapsIntersectMode::First))
                    .unwrap_or_else(|| caps.clone());
                q.set_result(&caps);
                true
            }
            QueryView::Seeking(ref mut q) => {
                q.set(
                    false,
                    gst::GenericFormattedValue::Time(0.into()),
                    gst::GenericFormattedValue::Time(0.into()),
                );
                true
            }
            _ => false,
        };

        if ret {
            gst_log!(self.cat, obj: pad, "Handled query {:?}", query);
        } else {
            gst_log!(self.cat, obj: pad, "Didn't handle query {:?}", query);
        }
        ret
        // pad.query_default(Some(element), query)
    }

    fn prepare(&self, element: &gst::Element) -> Result<(), gst::ErrorMessage> {
        gst_debug!(self.cat, obj: element, "Preparing");
        gst_debug!(self.cat, obj: element, "Prepared");
        Ok(())
    }

    fn unprepare(&self, element: &gst::Element) -> Result<(), ()> {
        gst_debug!(self.cat, obj: element, "Unpreparing");
        gst_debug!(self.cat, obj: element, "Unprepared");
        Ok(())
    }

    fn start(&self, element: &gst::Element) -> Result<(), ()> {
        gst_debug!(self.cat, obj: element, "Starting");
        gst_debug!(self.cat, obj: element, "Started");
        Ok(())
    }

    fn stop(&self, element: &gst::Element) -> Result<(), ()> {
        gst_debug!(self.cat, obj: element, "Stopping");
        gst_debug!(self.cat, obj: element, "Stopped");
        Ok(())
    }

    // The chain function is the function in which all data processing takes place.
    fn sink_chain(
        &self,
        pad: &gst::Pad,
        element: &gst::Element,
        inbuf: gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst_trace!(self.cat, obj: pad, "Handling buffer {:?}", inbuf);

        // Push buffer to zipper
        let zipper = &mut self.zipper.lock().unwrap();
        {
            let sink_pads = self.sink_pads.lock().unwrap();
            let info = sink_pads.get(pad).unwrap();
            zipper.push(inbuf, info.idx);
            gst_trace!(self.cat, obj: pad, "Pushed buffer to zipper {:?}", &info);
        };

        // Check if zipper can zip, process and push to srcs
        if let Some(buffers) = zipper.try_zip() {
            gst_trace!(self.cat, obj: pad, "Check if zipper can zip, process and push to srcs {:?}", buffers);
            let num_sink_pads = self.sink_pads.lock().unwrap().len();
            let num_src_pads = self.src_pads.lock().unwrap().len();

            // Prepare buffer inputs and outputs
            let mut outbufs: Vec<gst::Buffer> = Vec::new();
            outbufs.resize_with(num_src_pads, || gst::Buffer::new());
            assert_eq!(
                num_sink_pads,
                buffers.len(),
                "Num of sink pads don't match IN buffers"
            );
            assert_eq!(
                num_src_pads,
                outbufs.len(),
                "Num of src pads don't match OUT buffers"
            );

            // Process buffers
            let mut state = self.state.lock().unwrap();
            T::process(&mut state.processor, &buffers, &mut outbufs).unwrap();

            // Send processed buffers through src pads
            let src_pads = self.src_pads.lock().unwrap();
            for _ in 0..num_src_pads {
                let outbuf = outbufs.pop().unwrap();
                let idx = outbufs.len();
                let (pad, info) = src_pads.iter().find(|(_, info)| info.idx == idx).unwrap();
                let res = pad.push(outbuf);
                gst_trace!(
                    self.cat,
                    obj: element,
                    "Pushing buffer for stream {:?} returned {:?}",
                    info.name,
                    res
                );
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }
}

lazy_static! {
    static ref OBJSUBCLASS_TYPES: Mutex<HashSet<&'static str>> = Mutex::new(HashSet::new());
}

impl<T> ObjectSubclass for Cata<T>
where
    T: 'static + Send + Default + Process + CapsDef + Registry,
{
    const NAME: &'static str = T::NAME;
    type ParentType = gst::Element;
    type Instance = gst::subclass::ElementInstanceStruct<Self>;
    type Class = subclass::simple::ClassStruct<Self>;

    // Our own glib_object_subclass!(); implementation
    fn type_data() -> std::ptr::NonNull<glib::subclass::TypeData> {
        T::type_data()
    }

    // Custom implementation for keeping track of derived types.
    // glib_object_subclass macro doesn't work with derived types.
    fn get_type() -> glib::Type {
        {
            let gtypes = &mut OBJSUBCLASS_TYPES.lock().unwrap();
            if !gtypes.contains(Self::NAME) {
                gtypes.insert(Self::NAME);
                glib::subclass::register_type::<Self>();
            };
        }
        unsafe {
            let data = Self::type_data();
            let type_ = data.as_ref().get_type();
            assert_ne!(type_, glib::Type::Invalid);
            type_
        }
    }

    fn class_init(klass: &mut subclass::simple::ClassStruct<Self>) {
        klass.set_metadata(
            "Catamorphism processor",
            "Cata/Aggregator",
            "Process n-to-n streams",
            "Simbotic",
        );

        // Create sink templates
        for caps in &T::caps_def().0 {
            let pad_template = gst::PadTemplate::new(
                caps.name,
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps.caps,
            )
            .unwrap();
            klass.add_pad_template(pad_template);
        }

        // Create source templates
        for caps in &T::caps_def().1 {
            let pad_template = gst::PadTemplate::new(
                caps.name,
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps.caps,
            )
            .unwrap();
            klass.add_pad_template(pad_template);
        }

        // Install all our properties
        klass.install_properties(T::properties());
    }

    fn new_with_class(klass: &subclass::simple::ClassStruct<Self>) -> Self {
        let mut sink_pads: PadMap = PadMap::new();
        let mut src_pads: PadMap = PadMap::new();

        // Setup sink pads
        for (idx, caps) in T::caps_def().0.iter().enumerate() {
            let templ = klass.get_pad_template(caps.name).unwrap();
            let pad = gst::Pad::new_from_template(&templ, Some(caps.name));

            // Callback for handling downstream events
            // https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html?gi-language=c
            pad.set_event_function(|pad, parent, event| {
                Self::catch_panic_pad_function(
                    parent,
                    || false,
                    |cata, element| cata.cata_event(pad, element, event),
                )
            });

            // Callback for handling downstream queries
            // https://gstreamer.freedesktop.org/documentation/plugin-development/basics/queryfn.html?gi-language=c
            pad.set_query_function(|pad, parent, query| {
                Self::catch_panic_pad_function(
                    parent,
                    || false,
                    |cata, element| cata.cata_query(pad, element, query),
                )
            });

            // Callback for handling data processing
            // https://gstreamer.freedesktop.org/documentation/plugin-development/basics/chainfn.html?gi-language=c
            pad.set_chain_function(|pad, parent, buffer| {
                Self::catch_panic_pad_function(
                    parent,
                    || Err(gst::FlowError::Error),
                    |cata, element| cata.sink_chain(pad, element, buffer),
                )
            });

            sink_pads.insert(
                pad,
                PadInfo {
                    name: caps.name,
                    idx,
                    caps: caps.caps.clone(),
                },
            );
        }

        // Setup source pads
        for (idx, caps) in T::caps_def().1.iter().enumerate() {
            let templ = klass.get_pad_template(caps.name).unwrap();
            let pad = gst::Pad::new_from_template(&templ, Some(caps.name));

            // Callback for handling upstream events
            // https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html?gi-language=c
            pad.set_event_function(|pad, parent, event| {
                Self::catch_panic_pad_function(
                    parent,
                    || false,
                    |cata, element| cata.cata_event(pad, element, event),
                )
            });

            // Callback for handling downstream queries
            // https://gstreamer.freedesktop.org/documentation/plugin-development/basics/queryfn.html?gi-language=c
            pad.set_query_function(|pad, parent, query| {
                Self::catch_panic_pad_function(
                    parent,
                    || false,
                    |cata, element| cata.cata_query(pad, element, query),
                )
            });

            src_pads.insert(
                pad,
                PadInfo {
                    name: caps.name,
                    idx,
                    caps: caps.caps.clone(),
                },
            );
        }

        // Setup buffer zipper
        let zipper = Zipper::with_size(sink_pads.len());

        // Create new instance of Cata
        Self {
            cat: gst::DebugCategory::new(
                T::DEBUG_CATEGORY,
                gst::DebugColorFlags::empty(),
                Some("Cata"),
            ),
            sink_pads: Mutex::new(sink_pads),
            src_pads: Mutex::new(src_pads),
            zipper: Mutex::new(zipper),
            state: Mutex::new(Default::default()),
        }
    }
}

impl<T> ObjectImpl for Cata<T>
where
    T: 'static + Send + Default + Process + CapsDef + Registry,
{
    glib_object_impl!();

    // Construct pads and add to element
    fn constructed(&self, obj: &glib::Object) {
        self.parent_constructed(obj);
        let element = obj.downcast_ref::<gst::Element>().unwrap();
        for (pad, info) in self.src_pads.lock().unwrap().iter() {
            pad.set_active(true).unwrap();
            let mut caps = info.caps.clone();
            caps.fixate();
            pad.push_event(gst::Event::new_caps(&caps).build());
            // TODO: proper segment handling
            let segment = gst::FormattedSegment::<gst::ClockTime>::default();
            pad.push_event(gst::Event::new_segment(&segment).build());
            element.add_pad(pad).unwrap();
        }
        for (pad, _info) in self.sink_pads.lock().unwrap().iter() {
            element.add_pad(pad).unwrap();
        }
    }

    // Called whenever a value of a property is changed. It can be called
    // at any time from any thread.
    fn set_property(&self, _obj: &glib::Object, id: usize, value: &glib::Value) {
        let prop = &T::properties()[id];
        let mut state = self.state.lock().unwrap();
        T::set_property(&mut state.processor, prop, value);
    }

    // Called whenever a value of a property is read. It can be called
    // at any time from any thread.
    fn get_property(&self, _obj: &glib::Object, id: usize) -> Result<glib::Value, ()> {
        let prop = &T::properties()[id];
        match *prop {
            _ => unimplemented!(),
        }
    }
}

impl<T> ElementImpl for Cata<T>
where
    T: 'static + Send + Default + Process + CapsDef + Registry,
{
    fn change_state(
        &self,
        element: &gst::Element,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        gst_trace!(self.cat, obj: element, "Changing state {:?}", transition);

        match transition {
            gst::StateChange::NullToReady => {
                self.prepare(element).map_err(|err| {
                    element.post_error_message(&err);
                    gst::StateChangeError
                })?;
            }
            gst::StateChange::PlayingToPaused => {
                self.stop(element).map_err(|_| gst::StateChangeError)?;
            }
            gst::StateChange::ReadyToNull => {
                self.unprepare(element).map_err(|_| gst::StateChangeError)?;
            }
            _ => (),
        }

        let mut success = self.parent_change_state(element, transition)?;

        match transition {
            gst::StateChange::ReadyToPaused => {
                success = gst::StateChangeSuccess::NoPreroll;
            }
            gst::StateChange::PausedToPlaying => {
                self.start(element).map_err(|_| gst::StateChangeError)?;
            }
            gst::StateChange::PausedToReady => {}
            _ => (),
        }

        Ok(success)
    }
}

pub fn register<T: 'static + Send + Default + Registry + Process + CapsDef>(
    plugin: &gst::Plugin,
) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        T::NAME,
        gst::Rank::None,
        Cata::<T>::get_type(),
    )
}
