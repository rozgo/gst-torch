use cgmath::{self, Deg, InnerSpace, Matrix3, Matrix4, Point3, Rad, Vector3};

use std::vec::Vec;

pub struct Model {
    pub graphics: Graphics,
    camera: Camera,
}

pub struct Graphics {
    pub texture_extent: wgpu::Extent3d,
    pub indices: Vec<u32>,
    pub index_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub color_texture: wgpu::Texture,
    pub color_texture_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_texture_view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub render_pipeline: wgpu::RenderPipeline,
}

struct Camera {
    eye: Point3<f32>,
    look: Point3<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Position(f32, f32, f32);

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Uniforms {
    world: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

impl Camera {
    fn view(&self) -> Matrix4<f32> {
        let direction = InnerSpace::normalize(self.look - self.eye);
        let up = Vector3::new(0.0, 1.0, 0.0);
        Matrix4::look_at_dir(self.eye, direction, up)
    }
}

pub fn model(device: &wgpu::Device, texture_extent: wgpu::Extent3d, indices: &Vec<u32>) -> Model {
    let msaa_samples = 1;

    let vs = include_bytes!("../../../assets/shaders/faceskin.vert.spv");
    let vs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap());

    let fs = include_bytes!("../../../assets/shaders/facedepth.frag.spv");
    let fs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs[..])).unwrap());

    let indices_bytes = indices_as_bytes(indices);
    let index_usage = wgpu::BufferUsage::INDEX;
    let index_buffer = device.create_buffer_with_data(indices_bytes, index_usage);

    let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: Some("facepose_color_texture"),
    });
    let color_texture_view = color_texture.create_default_view();

    let depth_format = wgpu::TextureFormat::Depth32Float;
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: depth_format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: Some("facepose_depth_texture"),
    });
    let depth_texture_view = depth_texture.create_default_view();

    let eye = Point3::new(60f32, 60f32, 50f32);
    let look = Point3::new(60f32, 60f32, 0.0f32);
    let camera = Camera { eye, look };

    let uniforms = create_uniforms([texture_extent.width, texture_extent.height], camera.view());
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
    let uniform_buffer = device.create_buffer_with_data(uniforms_bytes, usage);

    let bind_group_layout = create_bind_group_layout(device);
    let bind_group = create_bind_group(device, &bind_group_layout, &uniform_buffer);
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let render_pipeline = create_render_pipeline(
        device,
        &pipeline_layout,
        &vs_module,
        &fs_module,
        color_format,
        depth_format,
        msaa_samples,
    );

    let graphics = Graphics {
        texture_extent,
        indices: indices.clone(),
        index_buffer,
        uniform_buffer,
        color_texture,
        color_texture_view,
        depth_texture,
        depth_texture_view,
        bind_group,
        render_pipeline,
    };

    Model { graphics, camera }
}

pub fn update(_device: &wgpu::Device, time: gst::ClockTime, _model: &mut Model) {
    let _time = (time.nanoseconds().unwrap() as f64 / 1_000_000_000f64) as f32;
}

pub fn view(device: &wgpu::Device, vertices: &Vec<f32>, model: &Model) -> wgpu::CommandEncoder {
    let vertices_bytes = vertices_as_bytes(&vertices);
    let vertex_usage = wgpu::BufferUsage::VERTEX;
    let vertex_buffer = device.create_buffer_with_data(vertices_bytes, vertex_usage);

    let uniforms = create_uniforms(
        [
            model.graphics.texture_extent.width,
            model.graphics.texture_extent.height,
        ],
        model.camera.view(),
    );
    let uniforms_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsage::COPY_SRC;
    let new_uniform_buffer = device.create_buffer_with_data(uniforms_bytes, usage);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("facepose_encoder"),
    });

    encoder.copy_buffer_to_buffer(
        &new_uniform_buffer,
        0,
        &model.graphics.uniform_buffer,
        0,
        uniforms_size,
    );

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &model.graphics.color_texture_view,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.1,
                    g: 0.1,
                    b: 0.1,
                    a: 0.0,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: &model.graphics.depth_texture_view,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
        });

        render_pass.set_bind_group(0, &model.graphics.bind_group, &[]);
        render_pass.set_pipeline(&model.graphics.render_pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(model.graphics.index_buffer.slice(..));
        let index_range = 0..(model.graphics.indices.len()) as u32;
        let start_vertex = 0;
        let instance_range = 0..1;
        render_pass.draw_indexed(index_range, start_vertex, instance_range);
    }

    encoder
}

fn create_uniforms([w, h]: [u32; 2], view: Matrix4<f32>) -> Uniforms {
    let rotation = Matrix3::from_angle_y(Rad(0f32));
    let aspect_ratio = w as f32 / h as f32;
    let proj = cgmath::perspective(Deg(60f32), aspect_ratio, 0.1, 100.0);
    Uniforms {
        world: Matrix4::from(rotation).into(),
        view: view.into(),
        proj: proj.into(),
    }
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::UniformBuffer { dynamic: false },
        }],
        label: Some("facepose_bind_group_layout"),
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
        }],
        label: Some("facepose_bind_group"),
    })
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    let desc = wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    };
    device.create_pipeline_layout(&desc)
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_mod,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_mod,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Cw,
            cull_mode: wgpu::CullMode::Back,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: color_format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Position>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

pub mod bytes {
    pub unsafe fn from_slice<T>(slice: &[T]) -> &[u8]
    where
        T: Copy + Sized,
    {
        let len = slice.len() * std::mem::size_of::<T>();
        let ptr = slice.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, len)
    }

    pub unsafe fn from<T>(t: &T) -> &[u8]
    where
        T: Copy + Sized,
    {
        let len = std::mem::size_of::<T>();
        let ptr = t as *const T as *const u8;
        std::slice::from_raw_parts(ptr, len)
    }
}

fn indices_as_bytes(data: &[u32]) -> &[u8] {
    unsafe { bytes::from_slice(data) }
}

fn vertices_as_bytes(data: &[f32]) -> &[u8] {
    unsafe { bytes::from_slice(data) }
}

fn uniforms_as_bytes(uniforms: &Uniforms) -> &[u8] {
    unsafe { bytes::from(uniforms) }
}
