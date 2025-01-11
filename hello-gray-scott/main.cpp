#include <fmt/core.h>

#include <vector>
#include <fstream>
#include <sstream>

#include <glad/glad.h>
#include <EGL/egl.h>

#include <GLFW/glfw3.h>

#include "ShaderProgram.h"

// Set to true to use test data for the texture
bool USE_TEST_DATA = false;

// Set to true to enable fullscreen
bool FULLSCREEN = false;

// Gray Scott Reaction Diffusion Frid
const int WIDTH = 640, HEIGHT = 320;

GLFWwindow* gWindow = NULL;
const char* APP_TITLE = "Gray Scott - Compute Shader";

// Window dimensions
const int gWindowWidth = WIDTH;
const int gWindowHeight = HEIGHT;

// Fullscreen dimensions
int gWindowWidthFull = 1920;
int gWindowHeightFull = 1200;

bool gWireframe = false;

// Function prototypes
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height);
void showFPS(GLFWwindow* window);
bool initOpenGL();

// Read a compute shader to string
std::string fileToString(const std::string& filename);

// Statically allocated arrays for the simulation
float A1cpu[gWindowWidth * gWindowHeight];
float A2cpu[gWindowWidth * gWindowHeight];
float B1cpu[gWindowWidth * gWindowHeight];
float B2cpu[gWindowWidth * gWindowHeight];

// Testing texture data
GLuint testData[WIDTH * HEIGHT * 4];

void checkGLError(const char* functionName) {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		const char* error;
		switch (err) {
		case GL_INVALID_ENUM:
			error = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			error = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			error = "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			error = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			error = "GL_OUT_OF_MEMORY";
			break;
		default:
			error = "UNKNOWN_ERROR";
		}
		printf("OpenGL Error: %s in %s\n", error, functionName);
	}
}

void checkProgramLinking(GLuint program) {
	GLint success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		char log[512];
		glGetProgramInfoLog(program, 512, NULL, log);
		printf("Program Linking Error: %s\n", log);
	}
}

void checkShaderCompilation(GLuint shader) {
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[512];
		glGetShaderInfoLog(shader, 512, NULL, log);
		printf("Compute Shader Compilation Error: %s\n", log);
	}
}

int main(int argc, char **argv)
{
	initOpenGL();

	// Load the compute shader
	std::string csString = fileToString("shader/gray-scott.cs");
	const GLchar* csSourcePtr = csString.c_str();

	GLuint compute_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute_shader, 1, &csSourcePtr, NULL);
	glCompileShader(compute_shader);
	checkShaderCompilation(compute_shader);

	GLuint compute_program = glCreateProgram();
	glAttachShader(compute_program, compute_shader);
	glLinkProgram(compute_program);
	checkProgramLinking(compute_program);

	// Load the vertex and fragment shaders for rendering the results
	ShaderProgram shader;
	shader.loadShaders("shader/vert.glsl", "shader/frag.glsl");

	// Set up the vertices and texure coordinates for two quads (one rectangle)
	GLfloat vertices[] = {
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,		// Top left
		 1.0f,  1.0f, 0.0f,	1.0f, 1.0f, 	// Top right
		 1.0f, -1.0f, 0.0f,	1.0f, 0.0f, 	// Bottom right
		-1.0f, -1.0f, 0.0f,	0.0f, 0.0f  	// Bottom left 
	};

	GLuint indices[] = {
		0, 1, 2,  // First Triangle
		0, 2, 3   // Second Triangle
	};

	// 2. Set up buffers on the GPU
	GLuint VAO, VBO, IBO;

	glGenVertexArrays(1, &VAO);				// Tell OpenGL to create new Vertex Array Object
	glGenBuffers(1, &VBO);					// Generate an empty vertex buffer on the GPU
	glGenBuffers(1, &IBO);					// Create buffer space on the GPU for the index buffer

	glBindBuffer(GL_ARRAY_BUFFER, VBO);		// "bind" or set as the current buffer we are working with
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);	// copy the data from CPU to GPU

	glBindVertexArray(VAO);					// Make it the current one
	
	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), NULL);	// Define a layout for the first vertex buffer "0"
	glEnableVertexAttribArray(0);			// Enable the first attribute or attribute "0"

	// Texture attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(sizeof(float) * 3));	// Define a layout for the second vertex buffer "1"
	glEnableVertexAttribArray(1);			// Enable the second attribute or attribute "1"

	// Set up index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Unbind to make sure other code doesn't change it
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Create a texture to write to
	GLuint tex_output;

	glGenTextures(1, &tex_output);
	checkGLError("glGenTextures");

	glBindTexture(GL_TEXTURE_2D, tex_output);
	checkGLError("glBindTexture");

	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
	checkGLError("glTexStorage2D");

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	GLuint test_texture;
	glGenTextures(1, &test_texture);
	glBindTexture(GL_TEXTURE_2D, test_texture);

	if (USE_TEST_DATA)
	{
		// Initialize the test texture data
		for (int i = 0; i < WIDTH * HEIGHT; ++i) {
			if (i < (WIDTH * HEIGHT / 2))
				testData[i + 0 * (WIDTH * HEIGHT)] = 255; // R
			else
				testData[i + 0 * (WIDTH * HEIGHT)] = 0;   // R

			testData[i + 1 * (WIDTH * HEIGHT)] = 0;   // G
			testData[i + 2 * (WIDTH * HEIGHT)] = 0;   // B
			testData[i + 3 * (WIDTH * HEIGHT)] = 255; // A
		}

		// Allocate and upload the texture data
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, &testData);
		checkGLError("glGenTextures");

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	// Get the max work group count
	GLint work_grp_cnt[3];

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

	printf("max global (total) work group counts x:%i y:%i z:%i\n",
	work_grp_cnt[0], work_grp_cnt[1], work_grp_cnt[2]);

	// Get the max work group size
	GLint work_grp_size[3];

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

	printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n",
	work_grp_size[0], work_grp_size[1], work_grp_size[2]);

	// Get the max work group invocations
	GLint work_grp_inv;
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
	printf("max local work group invocations %i\n", work_grp_inv);

	GLint maxImageUnits;
	glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxImageUnits);
	printf("Max Image Units: %d\n", maxImageUnits);

	GLint maxTexSize;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
	printf("Max Texture Size: %d\n", maxTexSize);

	// Dynamically allocated arrays for the simulation
	//float* A1cpu = new float[gWindowWidth*gWindowHeight];
	//float* A2cpu = new float[gWindowWidth*gWindowHeight];
	//float* B1cpu = new float[gWindowWidth*gWindowHeight];
	//float* B2cpu = new float[gWindowWidth*gWindowHeight];

	// Initialize the simulation
    for(int x=0; x < gWindowWidth; x++)
	{
		for(int y=0; y < gWindowHeight; y++)
		{
			int idx = x + y * gWindowWidth;
			A1cpu[idx] = 1.0;
			A2cpu[idx] = 1.0;

			if (rand() / float(RAND_MAX) < 0.0021)
				B1cpu[idx] = 1.0;
			else 
				B1cpu[idx] = 0.0;

			B2cpu[idx] = 0.0;
		}
	}

	// Buffer object IDs
	GLuint A1, B1, A2, B2;

	// Generate buffer objects
    glGenBuffers(1, &A1);
    glGenBuffers(1, &B1);
    glGenBuffers(1, &A2);
    glGenBuffers(1, &B2);

	// Bind the buffer to a specific binding point
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, A1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * gWindowWidth * gWindowHeight, A1cpu, GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, A2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * gWindowWidth * gWindowHeight, A2cpu, GL_STATIC_DRAW);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, B1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * gWindowWidth * gWindowHeight, B1cpu, GL_STATIC_DRAW);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, B2);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * gWindowWidth * gWindowHeight, B2cpu, GL_STATIC_DRAW);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A2);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, B1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, B2);

    // Unbind the buffer (optional)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	int c = 1;
	while (glfwWindowShouldClose(gWindow) == 0) {
		// Vsync - comment this out if you want to disable vertical sync
		//glfwSwapInterval(0);

		showFPS(gWindow);

		{
			c = 1 - c;
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0 + c, A1);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0 + 1 - c, A2);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + c, B1);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + 1 - c, B2);

			// launch compute shaders!
			glUseProgram(compute_program);
			checkGLError("glUseProgram");

			glUniform1i(glGetUniformLocation(compute_program, "W"), WIDTH);
			checkGLError("glUniform1i");
			glUniform1i(glGetUniformLocation(compute_program, "H"), HEIGHT);
			checkGLError("glUniform1i");

			if (tex_output != 0) {
				glBindImageTexture(4, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
			}
			else {
				// Handle the error, e.g., log it or initialize the texture
				fmt::println("Error: tex_output is not initialized properly.");
			}
			glDispatchCompute(WIDTH / 10, HEIGHT / 10, 1);
		}
		
		// make sure writing to image has finished before read
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		{ 
			// normal drawing pass
			glClear(GL_COLOR_BUFFER_BIT);

			shader.use();

			glActiveTexture(GL_TEXTURE0);
			if (USE_TEST_DATA)
				glBindTexture(GL_TEXTURE_2D, test_texture);
			else
				glBindTexture(GL_TEXTURE_2D, tex_output);

			glUniform1i(glGetUniformLocation(shader.getProgram(), "screenTexture"), 0);

			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

			glBindVertexArray(0);
		}

		glfwSwapBuffers(gWindow);
		glfwPollEvents();
	}

	// Clean up
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &IBO);
	glDeleteVertexArrays(1, &VAO);

	glDeleteBuffers(1, &A1);
	glDeleteBuffers(1, &A2);
	glDeleteBuffers(1, &B1);
	glDeleteBuffers(1, &B2);

	shader.destroy();

	glfwTerminate();

	return 0;
}

std::string fileToString(const std::string& filename)
{
	std::stringstream ss;
	std::ifstream file;

	try
	{
		file.open(filename, std::ios::in);

		if (!file.fail())
		{
			// Using a std::stringstream is easier than looping through each line of the file
			ss << file.rdbuf();
		}

		file.close();
	}
	catch (std::exception ex)
	{
		fmt::println("Error reading shader {}!", filename);
	}

	return ss.str();
}

// Press ESC to close the window
// Press 1 to toggle wireframe mode
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	//if (key == GLFW_KEY_1 && action == GLFW_PRESS)
	//{
	//	gWireframe = !gWireframe;
	//	if (gWireframe)
	//		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//	else
	//		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//}
}

// Is called when the window is resized
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

bool initOpenGL()
{
	if (!glfwInit())
	{
		fmt::println("GLFW initialization failed");
		return false;
	}

	// Set the OpenGL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);	// forward compatible with newer versions of OpenGL as they become available but not backward compatible (it will not run on devices that do not support OpenGL 3.3

	// Create a window
	if (FULLSCREEN)
	{
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		gWindowWidthFull = mode->width;
		gWindowHeightFull = mode->height;
		gWindow = glfwCreateWindow(gWindowWidthFull, gWindowHeightFull, APP_TITLE, glfwGetPrimaryMonitor(), NULL);
	}
	else
		gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight, APP_TITLE, NULL, NULL);

	if (gWindow == NULL)
	{
		fmt::println("Failed to create GLFW window");
		glfwTerminate();
		return false;
	}

	// Make the window's context the current one
	glfwMakeContextCurrent(gWindow);

	EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (display == EGL_NO_DISPLAY) {
		printf("Failed to get EGL display\n");
		return -1;
	}

	if (!eglInitialize(display, NULL, NULL)) {
		printf("Failed to initialize EGL\n");
		return -1;
	}
	//gladLoadGL();
	if (!gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to initialize GLAD for OpenGL ES 3.2\n");
        return -1;
    }
	printf("Loaded OpenGL ES Version: %d.%d\n", GLVersion.major, GLVersion.minor);

	// Set the required callback functions
	glfwSetKeyCallback(gWindow, glfw_onKey);
	glfwSetFramebufferSizeCallback(gWindow, glfw_onFramebufferSize);

	glClearColor(0.23f, 0.38f, 0.47f, 1.0f);

	if (FULLSCREEN)
		glViewport(0, 0, gWindowWidthFull, gWindowHeightFull);
	else
		glViewport(0, 0, gWindowWidth, gWindowHeight);

	return true;
}

void showFPS(GLFWwindow* window) {
	static double previousSeconds = 0.0;
	static int frameCount = 0;
	double elapsedSeconds;
	double currentSeconds = glfwGetTime();

	elapsedSeconds = currentSeconds - previousSeconds;

	if (elapsedSeconds > 0.25) {
		previousSeconds = currentSeconds;
		double fps = (double)frameCount / elapsedSeconds;
		double msPerFrame = 1000.0 / fps;

		char title[80];
		std::snprintf(title, sizeof(title), "Gray Scott @ fps: %.2f, ms/frame: %.2f", fps, msPerFrame);
		glfwSetWindowTitle(window, title);

		frameCount = 0;
	}

	frameCount++;
}
