#version 430 core

out vec4 vColor; // Output color to the fragment shader

// SSBO for particle positions
layout(binding = 0) buffer ParticlePositions {
    vec2 positions[]; // Array of vec2 for particle positions
};

// SSBO for particle colors
layout( binding = 1) buffer ParticleColors {
    vec4 colors[]; // Array of vec4 for particle colors
};

void main() {
    // Use gl_VertexID to index into the SSBOs
    vec2 position = positions[gl_VertexID] * 2.0 - 1.0;
    vec4 color = colors[gl_VertexID];

    vColor = color;                    // Pass color to fragment shader
    gl_Position = vec4(position, 0.0, 1.0); // Convert to clip space
    gl_PointSize = 10.0;                // Set point size
}
