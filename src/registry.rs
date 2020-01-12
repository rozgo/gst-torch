pub trait Registry {
    const NAME: &'static str;
    const DEBUG_CATEGORY: &'static str;
    fn type_data() -> ::std::ptr::NonNull<glib::subclass::TypeData>;
}

macro_rules! register_typedata {
    () => {
        fn type_data() -> ::std::ptr::NonNull<glib::subclass::TypeData> {
            static mut DATA: glib::subclass::TypeData = glib::subclass::TypeData {
                type_: glib::Type::Invalid,
                parent_class: ::std::ptr::null_mut(),
                interface_data: ::std::ptr::null_mut(),
                private_offset: 0,
            };
            unsafe { ::std::ptr::NonNull::new_unchecked(&mut DATA) }
        }
    };
}

