#version 310 es
precision mediump float;

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main()
{ 
    FragColor = texture(screenTexture, TexCoords);
}
