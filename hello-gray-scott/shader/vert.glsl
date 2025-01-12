#version 320 es
precision mediump float;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;

out vec2 TexCoords;

void main()
{
    gl_Position = vec4(in_position, 1.0); 
    TexCoords = in_texcoord;
}
