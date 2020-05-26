// NOTE: This shader requires being manually compiled to SPIR-V in order to
// avoid having downstream users require building shaderc and compiling the
// shader themselves. If you update this shader, be sure to also re-compile it
// and update `vert.spv`. You can do so using `glslangValidator` with the
// following command: `glslangValidator -V shader.vert`

#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    vec3 pos = vec3(position.x, position.y, position.z);
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(pos, 1.0);
}
