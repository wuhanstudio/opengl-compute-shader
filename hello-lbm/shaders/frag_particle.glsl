#version 430 core

in vec4 vColor;       // Interpolated color from vertex shader
out vec4 FragColor;   // Final fragment color

void main() {
    FragColor = vColor; // Set fragment color
}
