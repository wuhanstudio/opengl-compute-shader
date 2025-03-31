//
//Standard Lattice Boltzmann BGK solver
//code: Maciej Matyka (http://www.matyka.pl/, maq@ift.uni.wroc.pls) 
//03.2015 (Wroclaw)
//34
//compile windows:
//    MSVisualC++ project included
//compile linux (Nicolas Delbosc):
//    g++ main.cpp -lglut -L /usr/lib64/ -lGLEW -lGLU LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
//let us know where does this end if anywhere, contact: maciej.matyka@gmail.com
//
//
// 1-fluid,0-boundary

// 01.2024 (Han Wu)
// 1. Replaced GLUT with GLFW
// 2. Replaced GLEW with GLAD
// 3. Replaced fixed pipeline with vert and frag shaders


#include <fmt/core.h>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ShaderProgram.h"

// Set to true to enable fullscreen
bool FULLSCREEN = false;

GLFWwindow* gWindow = NULL;
const char* APP_TITLE = "Hello LBM";

// Window dimensions
int gWindowWidth = 1280;
int gWindowHeight = 720;

const int NX = 640;		// solver grid resolution
const int NY = 360;
const int NUM_PARTICLE = 1000000;

// Fullscreen dimensions
const int gWindowWidthFull = 1920;
const int gWindowHeightFull = 1200;

bool gWireframe = false;

GLuint VAO, VBO;
ShaderProgram obstacleShader;
ShaderProgram particleShader;

std::vector<float> vertices;

void showFPS(GLFWwindow* window);
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_onMouse(GLFWwindow* window, int button, int action, int mods);
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height);

void init(void);
void init_shaders(void);
void init_buffers(void);

/*--------------------- Mouse ---------------------------------------------------------------------------*/
int mousedown = 0;
float xMouse, yMouse;

/*--------------------- LBM -----------------------------------------------------------------------------*/
#define NUMR 20
#define NUM_VECTORS 9	// lbm basis vectors (d2q9 model)

float fx = 1, fx2 = 1;
float fy = 0, fy2 = 0;

float angle = 0;				// for rotations of the body force vec
float force = -0.000007;		// body force magnitude
int c = 0;

/*--------------------- LBM State vector ----------------------------------------------------------------*/
GLuint c0_SSB;
GLuint c1_SSB;

GLuint cF_SSB;
GLuint cU_SSB;
GLuint cV_SSB;

int F_cpu[NX * NY];

/*--------------------- Particles -----------------------------------------------------------------------*/
float dt = 0.1;

GLuint col_SSB;
GLuint particles_SSB;

struct p
{
	float x, y;
};

struct col
{
	float r, g, b, a;
};

/*--------------------- Shader Programs ------------------------------------------------------------------*/
GLuint lbmCS_Program;
GLuint moveparticlesCS_Program;

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

/*--------------------- Generate buffers------------------------------------------------------------------*/
void GenerateSSB(GLuint& bufid, int width, int height, float a)
{
	glGenBuffers(1, &bufid);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufid);
	glBufferData(GL_SHADER_STORAGE_BUFFER, width * height * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	float* temp = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, width * height * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			temp[x + y * width] = a;
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

/*--------------------- Reset positions in particle buffers -----------------------------------------------*/
void resetparticles(void)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, particles_SSB);
	p* parGPU = (p*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, NUM_PARTICLE * sizeof(p), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	int i = 0;
	for (i = 0; i < NUM_PARTICLE; i++)
	{
		parGPU[i].x = (float)rand() / (float)RAND_MAX;
		parGPU[i].y = (float)rand() / (float)RAND_MAX;
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

/*--------------------- Update obstacle flags -------------------------------------------------------------*/
void updateObstacle(void)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cF_SSB);
	int* F_temp = (int*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, NX * NY * sizeof(int), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

	for (int x = 0; x < NX; x++)
		for (int y = 0; y < NY; y++)
		{
			int xx = x - (xMouse) * NX / 2.0;
			int yy = y - (yMouse) * NY / 2.0;
			int idx = x + y * NX;
			if (idx > NX * NY)	break;
			if (sqrt(float((xx - NX / 2) * (xx - NX / 2) + (yy - NY / 2) * (yy - NY / 2))) < NX / 14)
			{
				F_cpu[idx] = 0;
				F_temp[idx] = 0;
			}
			else
			{
				F_cpu[idx] = 1;
				F_temp[idx] = 1;
			}
		}

	for (int x = 0; x < NX; x++)
		F_temp[x + 0 * NX] = F_temp[x + (NY - 1) * NX] = 0;

	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	vertices.clear();
	for (int x = 0; x < NX; x++) {
		for (int y = 0; y < NY; y++) {
			int idx = x + y * NX;

			float x1 = static_cast<float>(x) / NX * 2.0 - 1.0;
			float y1 = static_cast<float>(y) / NY * 2.0 - 1.0;
			float dx = 1.0f / NX * 2.0;
			float dy = 1.0f / NY * 2.0;

			if (F_cpu[idx] == 0) {
				// Define quad vertices and color
				vertices.insert(vertices.end(),
					{
						x1, y1,
						x1 + dx, y1,
						x1 + dx, y1 + dy,
						x1 + dx, y1 + dy,
						x1, y1 + dy,
						x1, y1,
					});
			}
		}
	}
}

void init(void)
{
	int i;
	fx2 = fx; fy2 = fy;		// init force

	/*-------------------- Compute shaders programs etc. ----------------------------------------------------*/
	init_shaders();
	init_buffers();
}

void init_shaders(void)
{
	char log[2048];
	int len = 0;

	// Create the compute shader for LBM
	GLuint lbmCS_Shader;
	std::string csString = fileToString("shaders/lbm.cs");
	const GLchar* lbmCS_Source = csString.c_str();
	lbmCS_Shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(lbmCS_Shader, 1, &lbmCS_Source, NULL);
	glCompileShader(lbmCS_Shader);

	glGetShaderInfoLog(lbmCS_Shader, 12047, &len, log);
	log[len] = '\0';
	fmt::println("Shader compiled: {}", log);

	// Create the compute shader program
	lbmCS_Program = glCreateProgram();
	glAttachShader(lbmCS_Program, lbmCS_Shader);
	glLinkProgram(lbmCS_Program);
	glUseProgram(lbmCS_Program);
	glUniform1i(0, NX);
	glUniform1i(1, NY);
	glUseProgram(0);

	// Create the compute shader for moving particles
	GLuint moveparticlesCS_Shader;
	std::string csPString = fileToString("shaders/particles.cs");
	const GLchar* moveparticlesCS_Source = csPString.c_str();
	moveparticlesCS_Shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(moveparticlesCS_Shader, 1, &moveparticlesCS_Source, NULL);
	glCompileShader(moveparticlesCS_Shader);

	glGetShaderInfoLog(moveparticlesCS_Shader, 1023, &len, log);
	log[len] = '\0';
	fmt::println("Shader compiled: {}", log);

	// Create the compute shader program
	moveparticlesCS_Program = glCreateProgram();
	glAttachShader(moveparticlesCS_Program, moveparticlesCS_Shader);
	glLinkProgram(moveparticlesCS_Program);
	glUseProgram(moveparticlesCS_Program);
	glUniform1i(0, NX);
	glUniform1i(1, NY);
	glUseProgram(0);

	// Create VAO and VBOs
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Load the vertex and fragment shaders for rendering the results
	obstacleShader.loadShaders("shaders/vert.glsl", "shaders/frag.glsl");
	particleShader.loadShaders("shaders/vert_particle.glsl", "shaders/frag_particle.glsl");
}

void init_buffers(void)
{
	/*---------------------- Initialise LBM vector state as SSB on GPU --------------------------------------*/
	float w[] = { (4.0 / 9.0),(1.0 / 9.0),(1.0 / 9.0),(1.0 / 9.0),(1.0 / 9.0),(1.0 / 36.0),(1.0 / 36.0),(1.0 / 36.0),(1.0 / 36.0) };

	glGenBuffers(1, &c0_SSB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, c0_SSB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NX * NY * sizeof(float) * NUM_VECTORS, NULL, GL_STATIC_DRAW);
	float* temp = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, NX * NY * sizeof(float) * NUM_VECTORS, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	for (int k = 0; k < NUM_VECTORS; k++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
				temp[k + x * NUM_VECTORS + y * NX * NUM_VECTORS] = w[k];
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &c1_SSB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, c1_SSB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NX * NY * sizeof(float) * NUM_VECTORS, NULL, GL_STATIC_DRAW);
	temp = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, NX * NY * sizeof(float) * NUM_VECTORS, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	for (int k = 0; k < NUM_VECTORS; k++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
				temp[k + x * NUM_VECTORS + y * NX * NUM_VECTORS] = w[k];
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &cF_SSB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cF_SSB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NX * NY * sizeof(int), NULL, GL_STATIC_DRAW);
	updateObstacle();
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	GenerateSSB(cU_SSB, NX, NY, 0.0);
	GenerateSSB(cV_SSB, NX, NY, 0.0);

	// Generate particles
	glGenBuffers(1, &particles_SSB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, particles_SSB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NUM_PARTICLE * sizeof(p), NULL, GL_STATIC_DRAW);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	resetparticles();

	glGenBuffers(1, &col_SSB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, col_SSB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NUM_PARTICLE * sizeof(struct col), NULL, GL_STATIC_DRAW);
	struct col* colors = (struct col*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, NUM_PARTICLE * sizeof(struct col), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

	for (int i = 0; i < NUM_PARTICLE; i++)
	{
		float r = rand() / (float)RAND_MAX;
		float g = r;// rand() / (float)RAND_MAX;
		float b = r;// rand() / (float)RAND_MAX;
		colors[i].r = r;// *(float)i / (float)NUMP;
		colors[i].g = g;
		colors[i].b = b;
		colors[i].a = 0.2;
		i++;
	}

	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	/*---------------------- Some bindings ------------------------------------------------------------------*/
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cF_SSB);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cU_SSB);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cV_SSB);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, particles_SSB);
}

bool initOpenGL()
{
	// Intialize GLFW 
	if (!glfwInit())
	{
		fmt::println("GLFW initialization failed");
		return false;
	}

	// Set the OpenGL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);	// forward compatible with newer versions of OpenGL as they become available but not backward compatible (it will not run on devices that do not support OpenGL 3.3

	glfwWindowHint(GLFW_RED_BITS, 8);		// Red channel bits
	glfwWindowHint(GLFW_GREEN_BITS, 8);		// Green channel bits
	glfwWindowHint(GLFW_BLUE_BITS, 8);		// Blue channel bits
	glfwWindowHint(GLFW_ALPHA_BITS, 8);		// Alpha channel bits

	// Create a window
	if (FULLSCREEN)
		gWindow = glfwCreateWindow(gWindowWidthFull, gWindowHeightFull, APP_TITLE, glfwGetPrimaryMonitor(), NULL);
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
	glfwSetMouseButtonCallback(gWindow, glfw_onMouse);
	glfwSetFramebufferSizeCallback(gWindow, glfw_onFramebufferSize);
	glfwSetCursorPos(gWindow, gWindowWidth / 2.0, gWindowHeight / 2.0);

	if (FULLSCREEN)
		glViewport(0, 0, gWindowWidthFull, gWindowHeightFull);
	else
		glViewport(0, 0, gWindowWidth, gWindowHeight);

	init();

	return true;
}

void render(void)
{
	if (mousedown) {
		//glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		double lastMouseX, lastMouseY;
		// Get the current mouse cursor position delta
		glfwGetCursorPos(gWindow, &lastMouseX, &lastMouseY);

		if (FULLSCREEN) {
			xMouse = 2.0 * ((float)lastMouseX / (float)gWindowWidthFull - 0.5);
			yMouse = -2.0 * ((float)lastMouseY / (float)gWindowHeightFull - 0.5);
		}
		else
		{
			xMouse = 2.0 * ((float)lastMouseX / (float)gWindowWidth - 0.5);
			yMouse = -2.0 * ((float)lastMouseY / (float)gWindowHeight - 0.5);
		}
		updateObstacle();
	}

	// computation (!)
	for (int i = 0; i < NUMR; i++)
	{
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, c, c0_SSB);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1 - c, c1_SSB);
		c = 1 - c;
		glUseProgram(lbmCS_Program);
		glUniform1f(2, fx2 * force);				// set body force in the shader
		glUniform1f(3, fy2 * force);
		glDispatchCompute(NX / 10, NY / 10, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glUseProgram(0);
	}

	glUseProgram(moveparticlesCS_Program);
	glDispatchCompute(NUM_PARTICLE / 1000, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glUniform1f(2, dt);
	glUseProgram(0);

	// Render
	glClear(GL_COLOR_BUFFER_BIT);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_MULTISAMPLE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Bind VAO VBO
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Render obstacles
	obstacleShader.use();
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 2);
	glBindVertexArray(0);

	// Render particles
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particles_SSB);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, col_SSB);

	GLuint defaultVAO;
	glGenVertexArrays(1, &defaultVAO);
	glBindVertexArray(defaultVAO);

	particleShader.use();
	glDrawArrays(GL_POINTS, 0, NUM_PARTICLE); // Render particles
	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Swap the front and back buffers
	glfwSwapBuffers(gWindow);
	glfwPollEvents();
}

void glfw_onFramebufferSize(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
	gWindowWidth = width;
	gWindowHeight = height;
}

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

	if (key == GLFW_KEY_D && action == GLFW_PRESS) { dt = -dt; }
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) { resetparticles(); }
	if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS) { force *= (-1); }
	if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS) { force *= 0.98; }

	if (key == GLFW_KEY_R && action == GLFW_PRESS)
	{
		angle += 2 * 3.14 / 180.0;
		fx2 = fx * cos(angle) - fy * sin(angle);
		fy2 = fx * sin(angle) + fy * cos(angle);
	}
}

void glfw_onMouse(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (GLFW_PRESS == action)
			mousedown = 1;
		else if (GLFW_RELEASE == action)
			mousedown = 0;
	}
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
		std::snprintf(title, sizeof(title), "Hello LBM @ fps: %.2f, ms/frame: %.2f", fps, msPerFrame);
		glfwSetWindowTitle(window, title);

		frameCount = 0;
	}

	frameCount++;
}

/*--------------------- Main loop ---------------------------------------------------------------------------*/
int main(int argc, char** argv)
{

	if (!initOpenGL())
	{
		fmt::println("GLFW initialization failed");
		return -1;
	}

	while (!glfwWindowShouldClose(gWindow))
	{
		showFPS(gWindow);
		render();
	}

	glfwTerminate();
	return 0;
}
