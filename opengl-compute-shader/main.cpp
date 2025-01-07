#include <fmt/core.h>

#include <fstream>
#include <sstream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Set to true to enable fullscreen
bool FULLSCREEN = false;

GLFWwindow* gWindow = NULL;
const char* APP_TITLE = "ShaderToy";

// Window dimensions
const int gWindowWidth = 1280;
const int gWindowHeight = 720;

// Fullscreen dimensions
int gWindowWidthFull = 1920;
int gWindowHeightFull = 1200;

bool gWireframe = false;

// Function prototypes
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height);
void showFPS(GLFWwindow* window);
bool initOpenGL();

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
		fmt::println("Error reading shader filename!");
	}

	return ss.str();
}

int main(int argc, char **argv)
{
	initOpenGL();

	fmt::println("Initializing Compute Shader");
	
	std::string csString = fileToString("shader/gray-scott.cs");
	const GLchar* csSourcePtr = csString.c_str();

	GLuint compute_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute_shader, 1, &csSourcePtr, NULL);
	glCompileShader(compute_shader);

	GLuint compute_program = glCreateProgram();
	glAttachShader(compute_program, compute_shader);
	glLinkProgram(compute_program);

	// Set up the rectangle
	//1. Set up an array of vertices for a quad (2 triangls) with an index buffer data
	GLfloat vertices[] = {
		-1.0f,  1.0f, 0.0f,		// Top left
		 1.0f,  1.0f, 0.0f,		// Top right
		 1.0f, -1.0f, 0.0f,		// Bottom right
		-1.0f, -1.0f, 0.0f		// Bottom left 
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

	glBindVertexArray(VAO);					// Make it the current one
	glBindBuffer(GL_ARRAY_BUFFER, VBO);		// "bind" or set as the current buffer we are working with

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);	// copy the data from CPU to GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);	// Define a layout for the first vertex buffer "0"
	glEnableVertexAttribArray(0);			// Enable the first attribute or attribute "0"

	// Set up index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Unbind to make sure other code doesn't change it
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Create a texture to write to
	GLuint tex_output;
	int tex_w = 1280, tex_h = 720;

	glGenTextures(1, &tex_output);
	// glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_output);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex_w, tex_h, 0, GL_RGBA, GL_INT, NULL);
	glBindImageTexture(4, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

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

	float* A1cpu = new float[gWindowWidth*gWindowHeight];
	float* A2cpu = new float[gWindowWidth*gWindowHeight];
	float* B1cpu = new float[gWindowWidth*gWindowHeight];
	float* B2cpu = new float[gWindowWidth*gWindowHeight];

    // initialize
    for(int x=0; x < gWindowWidth; x++)
	{
		for(int y=0; y < gWindowHeight; y++)
		{
			int idx = x + y * gWindowWidth;
			A1cpu[idx] = 1.0;
			A2cpu[idx] = 1.0;
			if(rand()/float(RAND_MAX) < 0.000021)
				B1cpu[idx] = 1.0; else B1cpu[idx] = 0.0;
			B2cpu[idx] = 0.0;
		}
	}

	// Buffer object IDs
	GLuint A1, B1, A2, B2;
    GLuint bindingPoint = 0;

	// Generate buffer objects
    glGenBuffers(1, &A1);
    glGenBuffers(1, &B1);
    glGenBuffers(1, &A2);
    glGenBuffers(1, &B2);

	// Bind the buffer to a specific binding point
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, A1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(A1cpu), A1cpu, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, A1);

    bindingPoint = 1;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, A2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(A2cpu), A2cpu, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, A2);

    bindingPoint = 2;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, B1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(B1cpu), B1cpu, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, B1);

    bindingPoint = 3;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, B2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(B2cpu), B2cpu, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, B2);

    // Unbind the buffer (optional)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


	while (glfwWindowShouldClose(gWindow) == 0) {

		{ // launch compute shaders!
			glUseProgram(compute_program);
			glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
		}
		
		// make sure writing to image has finished before read
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		
		{ 
			// normal drawing pass
			glClear(GL_COLOR_BUFFER_BIT);
			glUseProgram(compute_program);
			glBindVertexArray(VAO);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, tex_output);
			// glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
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

	// shader.destroy();

	glfwTerminate();

	return 0;
}

// Press ESC to close the window
// Press 1 to toggle wireframe mode
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (key == GLFW_KEY_1 && action == GLFW_PRESS)
	{
		gWireframe = !gWireframe;
		if (gWireframe)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
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
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
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

	gladLoadGL();

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
		std::snprintf(title, sizeof(title), "Hello Shader @ fps: %.2f, ms/frame: %.2f", fps, msPerFrame);
		glfwSetWindowTitle(window, title);

		frameCount = 0;
	}

	frameCount++;
}
