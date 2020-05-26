// NOTE: This shader requires being manually compiled to SPIR-V in order to
// avoid having downstream users require building shaderc and compiling the
// shader themselves. If you update this shader, be sure to also re-compile it
// and update `frag.spv`. You can do so using `glslangValidator` with the
// following command: `glslangValidator -V shader.frag`

#version 450

// layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(1.0, -1.0, 1.0);

float near = 0.1; 
float far  = 100.0; 
  
float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main()
{             
    float depth = LinearizeDepth(gl_FragCoord.z) / far;
    f_color = vec4(vec3(pow(1.0 - depth, 2)), 1.0);
}