// Megan Reddy
// Monte Carlo Path Tracer
// To compile this file, type the following:
//   g++ pathtracer.cpp -lpng -lX11 -lpthread -o output
// To run, type the following:
//   ./output files/teapot.txt

#define cimg_use_png
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <math.h>
#include "CImg.h"

using namespace cimg_library;

#define EPSILON 1e-7
#define BIAS 1e-4

enum material_type{DIFFUSE, REFLECTIVE, REFRACTIVE};

// RAY AND VECTOR  

class Vec3 {
public:
    float x, y, z;
    Vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
    Vec3 operator+(const Vec3 &v2) const { return Vec3(x + v2.x, y + v2.y, z + v2.z); }
    Vec3 operator-(const Vec3 &v2) const { return Vec3(x - v2.x, y - v2.y, z - v2.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    bool operator==(const Vec3 &v2) const { return (x == v2.x && y == v2.y && z == v2.z); }
    bool operator!=(const Vec3 &v2) const { return !(*this == v2); }

    float mag() const {
        float squared_sum = x * x + y * y + z * z;
        return sqrt(squared_sum);
    }

    Vec3 norm() const { 
        return *this/mag();
    }
    
    float dot(const Vec3 &v2) const {
        return x * v2.x + y * v2.y + z * v2.z;
    }

    Vec3 cross(const Vec3 &v2) const {
        float rx = (y * v2.z) - (z * v2.y);
        float ry = (z * v2.x) - (x * v2.z);
        float rz = (x * v2.y) - (y * v2.x);
        return Vec3(rx, ry, rz);
    }

    Vec3 multVbyV(const Vec3 &v2) const {
        return Vec3(x * v2.x, y * v2.y, z * v2.z);
    }

    void print() {
        printf("(%f, %f, %f) \n", x, y, z);
    }
};

class Ray {
public:
    Vec3 origin;
    Vec3 direction;
    Ray(Vec3 o = 0, Vec3 d = 0) { origin = o; direction = d; }

    Vec3 point_at(float t) const { return origin + direction * t; }
};

// MATERIAL

class Material {
public:
    material_type type;
    Vec3 shininess, transparency;
    Vec3 diffuse_coefficient, reflective_coefficient, refractive_coefficient;
    float ior;
    Material() {
        type = DIFFUSE;
        shininess = Vec3(0, 0, 0);
        transparency = Vec3(0, 0, 0);
        ior = 1.458;
    }

    void calc_material_type() {
        if (transparency != Vec3(0, 0, 0)) {
            type = REFRACTIVE;
            refractive_coefficient = transparency;
        } else if (shininess != Vec3(0, 0, 0)) {
            type = REFLECTIVE;
            reflective_coefficient = shininess;
        } else {
            type = DIFFUSE;
        }
    }

    Vec3 calc_final_color(Vec3 &diffuse_color, Vec3 &reflective_color, Vec3 &refractive_color) {
        diffuse_coefficient.x = (1 - shininess.x) * (1 - transparency.x);
        diffuse_coefficient.y = (1 - shininess.y) * (1 - transparency.y);
        diffuse_coefficient.z = (1 - shininess.z) * (1 - transparency.z);
        diffuse_color = diffuse_color.multVbyV(diffuse_coefficient);

        refractive_coefficient.x = (1 - shininess.x) * (transparency.x);
        refractive_coefficient.y = (1 - shininess.y) * (transparency.y);
        refractive_coefficient.z = (1 - shininess.z) * (transparency.z);
        refractive_color = refractive_color.multVbyV(refractive_coefficient);

        reflective_color = reflective_color.multVbyV(reflective_coefficient);
        
        Vec3 final_color = diffuse_color + refractive_color + reflective_color;
        return final_color;
    }
};

// SCENE OBJECTS

class Intersection {
public:
    float distance;
    Vec3 point, normal;
    int edge;
    Intersection(float d = 0, Vec3 p = 0, Vec3 n = 0, int e = 0) { 
        distance = d; 
        point = p; 
        normal = n; 
        edge = e; 
    }

    void set_attributes(float t, const Ray &ray, Vec3 hit_point, Vec3 norm) {
        distance = t;
        point = hit_point;
        normal = norm;

        set_normal(ray);
        check_edge(ray);
    }

    void set_normal(const Ray &ray) {
        if (normal.dot(ray.direction) > 0) { normal = normal * -1; }
    }

    void check_edge(const Ray &ray) {
        if (normal.dot(ray.direction) < EPSILON) { 
            edge = 1; 
        } else {
            edge = 0;
        }
    }
};

class SceneObj {
public:
    Material material;
    Vec3 color;
    Vec3 emission;
    virtual bool intersect(const Ray &ray, Intersection &intersection) const = 0;
};

class Sphere : public SceneObj {
public:
    Vec3 center;
    float radius;
    Sphere(Vec3 c = 0, float r = 0) { center = c; radius = r; }
    bool intersect(const Ray &ray, Intersection &intersection) const override {
        Vec3 co = center - ray.origin;
        float e = co.dot(ray.direction);
        if (e < 0) return false; 

        float h2 = co.dot(co) - e*e;
        if (h2 > radius*radius) return false; 

        float f = sqrt(radius*radius - h2);

        // calculate distance
        float t = 0;
        float t1 = e - f;
        float t2 = e + f;

        if (t1 < 0 && t2 < 0) return false; 

        if (t1 < t2) {
            if (t1 < 0) {
                t = t2;
            } else {
                t = t1;
            }
        } else {
            t = t2;
        }

        Vec3 hit_point = ray.point_at(t);
        Vec3 normal = (hit_point - center).norm();

        // set intersection attributes (point, surface normal, etc.)
        intersection = Intersection();
        intersection.set_attributes(t, ray, hit_point, normal);

        return true;
    }
};

class RectXY : public SceneObj {
public:
    float x0, y0, x1, y1, k;
    RectXY(float x_0 = 0, float y_0 = 0, float x_1 = 0, float y_1 = 0, float k_0 = 0) { 
        x0 = x_0; 
        y0 = y_0; 
        x1 = x_1; 
        y1 = y_1;
        k = k_0; 
    }
    bool intersect(const Ray &ray, Intersection &intersection) const override {
        float ray_orig_z = ray.origin.z;
        float ray_dir_z = ray.direction.z;
        float t = (k - ray_orig_z) / ray_dir_z;

        Vec3 hit_point = ray.point_at(t);
        float x = hit_point.x;
        float y = hit_point.y;

        if (x < x0 || x > x1) return false; 
        if (y < y0 || y > y1) return false; 

        Vec3 normal = Vec3(0, 0, 1);

        // set intersection attributes (point, surface normal, etc.)
        intersection = Intersection();
        intersection.set_attributes(t, ray, hit_point, normal);
        
        return true;
    }
};

class Plane : public SceneObj {
public:
    Vec3 normal;
    float d;
    Plane(Vec3 n = 0, float d_0 = 0) { normal = n; d = d_0; }
    bool intersect(const Ray &ray, Intersection &intersection) const override {
        float denominator = normal.dot(ray.direction);

        if (denominator > 1e-13) {
            float t = d / denominator;
            if (t > 0) {
                Vec3 hit_point = ray.point_at(t);
                Vec3 surface_normal = normal.norm();

                // set intersection attributes (point, surface normal, etc.)
                intersection = Intersection();
                intersection.set_attributes(t, ray, hit_point, surface_normal);

                return true;
            }
        } 

        return false;
    }
};

class Triangle : public SceneObj {
public:
    Vec3 v0, v1, v2;
    Vec3 e1, e2;
    Vec3 normal;
    Triangle(Vec3 v_0 = 0, Vec3 v_1 = 0, Vec3 v_2 = 0, Vec3 e_1 = 0, Vec3 e_2 = 0, Vec3 norm = 0) { 
        v0 = v_0; 
        v1 = v_1;
        v2 = v_2;
        e1 = e_1;
        e2 = e_2;
        normal = norm;
    }
    bool intersect(const Ray &ray, Intersection &intersection) const override {
        // Step 1: ray-plane intersection
        float cos_theta = normal.dot(ray.direction);
        if (cos_theta == 0) return false;

        // Distance from origin to plane
        float d = normal.dot(v0);
        float t = 0.0;
        Vec3 o = ray.origin;
        if (o != Vec3(0, 0, 0)) { // for rays not originating from origin
            Vec3 x = v0 - o;
            t = (x.dot(normal)) / cos_theta;
        } else {
            t = (normal.dot(o) + d) / cos_theta;
        }
        if (t < 0.0001) return false;

        // Calculate hit point and surface normal
        Vec3 hit_point = ray.point_at(t);
        Vec3 surface_normal = normal.norm();

        // Step 2: check if in triangle
        Vec3 x1 = hit_point - v0;
        float b1 = e1.dot(x1);
        if (b1 < 0 || b1 > 1) return false;

        Vec3 x2 = hit_point - v1;
        float b2 = e2.dot(x2);
        if (b2 < 0 || b2 > 1) return false;

        if ((b1 + b2) > 1) return false;

        // set intersection attributes (point, surface normal, etc.)
        intersection = Intersection();
        intersection.set_attributes(t, ray, hit_point, surface_normal);

        return true;
    }
};

// SCENE LIGHTS

class Light {
public:
    Vec3 position;
    Vec3 color;
    virtual Vec3 direction(const Intersection &intersection) const = 0;
    virtual double distance(const Intersection &intersection) const = 0;
    virtual double intensity(const Intersection &intersection) const = 0;
};

class Sun : public Light {
public:
    Sun(Vec3 p = 0, Vec3 c = 0) { position = p; color = c; }
    Vec3 direction(const Intersection &intersection) const override {
        return position.norm();
    }

    double distance(const Intersection &intersection) const override {
        return std::numeric_limits<double>::max();
    }

    double intensity(const Intersection &intersection) const override {
        return 1;
    }
};

class Bulb : public Light {
public:
    Bulb(Vec3 p = 0, Vec3 c = 0) { position = p; color = c; }
    Vec3 direction(const Intersection &intersection) const override {
        return (position - intersection.point).norm();
    }

    double distance(const Intersection &intersection) const override {
        return (position - intersection.point).mag();
    }

    double intensity(const Intersection &intersection) const override {
        double d = distance(intersection);
        return 1 / (d*d);
    }
};

// CAMERA

class Camera {
public:
    Vec3 eye, forward, right, up;
    Camera() { 
        eye = Vec3(0, 0, 0); 
        forward = Vec3(0, 0, -1); 
        right = Vec3(1, 0, 0); 
        up = Vec3(0, 1, 0); 
    }

    Ray get_ray(int x, int y, int width, int height) {
        Ray ray;

        float denominator = std::max(width, height);
        float sx = (2 * x - width) / denominator;
        float sy = (height - 2 * y) / denominator;

        Vec3 sr_su = (right * sx) + (up * sy);
        ray.direction = (forward + sr_su).norm();
        ray.origin = eye;

        return ray;
    }
    
    void set_eye(std::vector<std::string> &tokens) { 
        eye.x = std::stof(tokens.at(1));
        eye.y = std::stof(tokens.at(2));
        eye.z = std::stof(tokens.at(3));
    }
};

// SCENE

class Scene {
public:
    int width, height, depth, spectrum; 
    std::string filename;
    CImg<float> img;
    Vec3 black, gray, white;
    float gamma_val;
    bool global_illumination;
    int secondary_rays, bounces;
    Material curr_material;
    std::vector<Vec3> vertex_list;
    std::vector<Vec3> color_list;
    std::vector<Vec3> emission_values;
    std::vector<std::shared_ptr<Light>> scene_lights; 
    std::vector<std::shared_ptr<SceneObj>> scene_objects;  
    Camera cam;
    Scene() { 
        width = 0; 
        height = 0; 
        depth = 1;
        spectrum = 4;
        black = Vec3(0, 0, 0); 
        gray = Vec3(0.5, 0.5, 0.5); 
        white = Vec3(1, 1, 1); 
        gamma_val = 1.0;
        global_illumination = false;
        secondary_rays = 1;
        bounces = 3;
        cam = Camera();
        curr_material = Material();
    }
};

// SCENE FILE READER

class SceneReader {
public:
    Scene scene;
    SceneReader() { scene = Scene(); }

    // FUNCTIONS TO PROCESS KEYWORDS THAT APPEAR IN SCENE FILES
    void sphere(std::vector<std::string> &tokens) {
        float x = std::stof(tokens.at(1));
        float y = std::stof(tokens.at(2));
        float z = std::stof(tokens.at(3));
        Vec3 center = Vec3(x, y, z);
        float radius = std::stof(tokens.at(4));

        // Create new sphere and add to render list
        auto new_sphere = std::make_shared<Sphere>(center, radius);
        new_sphere->material = scene.curr_material;
        new_sphere->color = scene.color_list.back();
        new_sphere->emission = scene.emission_values.back();
        scene.scene_objects.push_back(new_sphere);
    }

    void rectxy(std::vector<std::string> &tokens) {
        float x0 = std::stof(tokens.at(1));
        float y0 = std::stof(tokens.at(2));
        float x1 = std::stof(tokens.at(3));
        float y1 = std::stof(tokens.at(4));
        float k = std::stof(tokens.at(5));

        // Create new rectangle and add to render list
        auto new_rect = std::make_shared<RectXY>(x0, y0, x1, y1, k);
        new_rect->material = scene.curr_material;
        new_rect->color = scene.color_list.back();
        new_rect->emission = scene.emission_values.back();
        scene.scene_objects.push_back(new_rect);
    }

    void plane(std::vector<std::string> &tokens) {
        float a = std::stof(tokens.at(1));
        float b = std::stof(tokens.at(2));
        float c = std::stof(tokens.at(3));
        Vec3 normal = Vec3(-a, -b, -c);
        float d = std::stof(tokens.at(4));

        // Create new plane and add to render list
        auto new_plane = std::make_shared<Plane>(normal, d);
        new_plane->material = scene.curr_material;
        new_plane->color = scene.color_list.back();
        new_plane->emission = scene.emission_values.back();
        scene.scene_objects.push_back(new_plane);
    }

    void triangle(std::vector<std::string> &tokens) {
        int i0 = std::stoi(tokens.at(1));
        int i1 = std::stoi(tokens.at(2));
        int i2 = std::stoi(tokens.at(3));
        std::vector<int> vertices{i0, i1, i2};
        std::vector<int> indices;
        get_vertices(scene.vertex_list.size(), vertices, indices);

        std::vector<Vec3> final_vertices;
        for (int i : indices) {
            final_vertices.push_back(scene.vertex_list[i]);
        }

        Vec3 normal, e1, e2;
        find_vectors(final_vertices, normal, e1, e2);

        // Create new triangle and add to render list
        auto new_triangle = std::make_shared<Triangle>(final_vertices.at(0), final_vertices.at(1), final_vertices.at(2), e1, e2, normal);
        new_triangle->material = scene.curr_material;
        new_triangle->color = scene.color_list.back();
        new_triangle->emission = scene.emission_values.back();
        scene.scene_objects.push_back(new_triangle);
    }

    void get_vertices(int size, std::vector<int> &vertices, std::vector<int> &indices) {
        // Get correct vertices from scene's vertex list
        for (int vertex : vertices) {
            int curr;
            if (vertex > 0) {
                curr = vertex - 1;
            } else if (vertex < 0) {
                curr = size + vertex;
            }
            indices.push_back(curr);
        }
    }

    void find_vectors(std::vector<Vec3> final_vertices, Vec3 &normal, Vec3 &e1, Vec3 &e2) {
        Vec3 v0 = final_vertices.at(0);
        Vec3 v1 = final_vertices.at(1);
        Vec3 v2 = final_vertices.at(2);
        
        // Find normal of triangle
        Vec3 v1v0 = v1 - v0;
        Vec3 v2v0 = v2 - v0;
        Vec3 v0v1 = v0 - v1;
        Vec3 v2v1 = v2 - v1;

        normal = v1v0.cross(v2v0);

        Vec3 e1_num = v2v0.cross(normal);
        float e1_den = e1_num.dot(v1v0);
        e1 = e1_num / e1_den;

        Vec3 e2_num = v2v1.cross(normal);
        float e2_den = e2_num.dot(v0v1);
        e2 = e2_num / e2_den;
    }

    void xyz(std::vector<std::string> &tokens) {
        float vx = std::stof(tokens.at(1));
        float vy = std::stof(tokens.at(2));
        float vz = std::stof(tokens.at(3));
        Vec3 new_vertex = Vec3(vx, vy, vz);
        
        scene.vertex_list.push_back(new_vertex);
    }

    void color(std::vector<std::string> &tokens) {
        float r = std::stof(tokens.at(1));
        float g = std::stof(tokens.at(2));
        float b = std::stof(tokens.at(3));
        Vec3 new_color = Vec3(r, g, b);
        
        scene.color_list.push_back(new_color);
    }

    void emission(std::vector<std::string> &tokens) {
        float e1 = std::stof(tokens.at(1));
        float e2 = std::stof(tokens.at(2));
        float e3 = std::stof(tokens.at(3));
        Vec3 new_emission = Vec3(e1, e2, e3);
        
        scene.emission_values.push_back(new_emission);
    }

    void add_shininess(std::vector<std::string> &tokens) {
        Vec3 new_shininess;
        if (tokens.size() > 2) {
            new_shininess.x = std::stof(tokens.at(1));
            new_shininess.y = std::stof(tokens.at(2));
            new_shininess.z = std::stof(tokens.at(3));
        } else {
            float s_constant = std::stof(tokens.at(1));
            new_shininess = Vec3(s_constant, s_constant, s_constant);
        }
        scene.curr_material.shininess = new_shininess;
        scene.curr_material.calc_material_type();
    }

    void add_transparency(std::vector<std::string> &tokens) {
        Vec3 new_transparency;
        if (tokens.size() > 2) {
            new_transparency.x = std::stof(tokens.at(1));
            new_transparency.y = std::stof(tokens.at(2));
            new_transparency.z = std::stof(tokens.at(3));
        } else {
            float t_constant = std::stof(tokens.at(1));
            new_transparency = Vec3(t_constant, t_constant, t_constant);
        }
        scene.curr_material.transparency = new_transparency;
        scene.curr_material.calc_material_type();
    }

    void sun(std::vector<std::string> &tokens) {
        float x = std::stof(tokens.at(1));
        float y = std::stof(tokens.at(2));
        float z = std::stof(tokens.at(3));
        Vec3 direction = Vec3(x, y, z);
        Vec3 color = scene.color_list.back();

        // Create new sun light and add to scene lights
        auto new_light = std::make_shared<Sun>(direction, color);
        scene.scene_lights.push_back(new_light);
    }

    void bulb(std::vector<std::string> &tokens) {
        float x = std::stof(tokens.at(1));
        float y = std::stof(tokens.at(2));
        float z = std::stof(tokens.at(3));
        Vec3 position = Vec3(x, y, z);
        Vec3 color = scene.color_list.back();

        // Create new bulb light and add to scene lights
        auto new_light = std::make_shared<Bulb>(position, color);
        scene.scene_lights.push_back(new_light);
    }

    void gi(std::vector<std::string> &tokens) {
        scene.global_illumination = true;
    }

    void spp(std::vector<std::string> &tokens) {
        scene.secondary_rays = std::stoi(tokens.at(1));
    }

    void gamma(std::vector<std::string> &tokens) {
        scene.gamma_val = std::stof(tokens.at(1));
    }

    void create_png(std::vector<std::string> &tokens) {
        scene.width = std::stoi(tokens.at(1));
        scene.height = std::stoi(tokens.at(2));
        scene.filename = tokens.at(3);

        // Create image where each pixel is a float and set all pixels to 0
        scene.img.assign(scene.width, scene.height, scene.depth, scene.spectrum, 0); 
        // Set default color and emission values
        scene.color_list.push_back(Vec3(1, 1, 1));
        scene.emission_values.push_back(Vec3(0, 0, 0));
    }

    int process_tokens(std::vector<std::string> &tokens) {
        std::string keyword = tokens.at(0);

        // Library of keywords
        if (keyword == "png") {
            create_png(tokens);
        } else if (keyword == "sphere") {
            sphere(tokens);
        } else if (keyword == "rectxy") {
            rectxy(tokens);
        } else if (keyword == "plane") {
            plane(tokens);
        } else if (keyword == "trif" || keyword == "f") {
            triangle(tokens);
        } else if (keyword == "xyz" || keyword == "v") {
            xyz(tokens);
        } else if (keyword == "color") {
            color(tokens);
        } else if (keyword == "emission") {
            emission(tokens);
        } else if (keyword == "shininess") {
            add_shininess(tokens);
        } else if (keyword == "transparency") {
            add_transparency(tokens);
        } else if (keyword == "sun") {
            sun(tokens);
        } else if (keyword == "bulb") {
            bulb(tokens);
        } else if (keyword == "gi") {
            gi(tokens);
        } else if (keyword == "spp") {
            spp(tokens);
        } else if (keyword == "gamma") {
            gamma(tokens);
        } else if (keyword == "eye") {
            scene.cam.set_eye(tokens);
        }
        return 0;
    }

    void process_scene_file(std::string scene_file) {
        std::ifstream stream(scene_file); 
        std::string line;

        // Process each line
        while (std::getline(stream, line)) {
            std::vector<std::string> tokens;
            // Skip blank lines 
            if (line.size() == 1) {
                continue;
            }
            std::string token;
            std::istringstream s(line);
            while (s >> token) {
                tokens.push_back(token);
            }
            process_tokens(tokens);
        }
    }
};

// UTILITIES

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0, 1.0);

float random_number() {
    return distribution(generator);
}

float clamp(float color, float min, float max) {
    float c = color;
    if (c >= max) {
        c = max;
    } else if (c <= min) {
        c = min;
    }
    return c;
}

void write_color(CImg<float> &img, int x, int y, Vec3 color, float gamma) {
    float r = pow(color.x, (1/gamma));
    float g = pow(color.y, (1/gamma));
    float b = pow(color.z, (1/gamma));

    img(x, y, 0, 0) = clamp(r, 0, 1) * 255;
    img(x, y, 0, 1) = clamp(g, 0, 1) * 255;
    img(x, y, 0, 2) = clamp(b, 0, 1) * 255;
    img(x, y, 0, 3) = 255;
}

// PATH TRACING 

bool get_intersection(Ray &ray, std::shared_ptr<SceneObj> &draw_object, Intersection &hit, Scene &scene) {
    float min_distance = std::numeric_limits<float>::max(); 

    bool intersect = false;

    for (int i=0; i < scene.scene_objects.size(); i++) {
        Intersection intersection;
        bool found_intersection = scene.scene_objects.at(i)->intersect(ray, intersection);
        if (found_intersection) {  
            intersect = true;
            float t = intersection.distance;
            if (t < min_distance) {
                draw_object = scene.scene_objects.at(i);
                hit = intersection;
                min_distance = t;
            }
        }
    }
    return intersect;
}

int calc_shadows(Ray &shadow_ray, std::shared_ptr<Light> &light, Intersection &hit, Scene &scene) {
    double light_distance = light->distance(hit);
    for (int i=0; i < scene.scene_objects.size(); i++) {
        auto shadow_object = scene.scene_objects.at(i);
        Intersection shadow_hit;
        bool found_intersection = scene.scene_objects.at(i)->intersect(shadow_ray, shadow_hit);
        if (found_intersection) {
            if (shadow_hit.distance < light_distance) {
                return 0;
            }
        }
    }
    return 1;
}

Vec3 sample_hemisphere(float r1, float r2) {
    // Get random direction in hemisphere around point
    float r = sqrt(1 - r1*r1);
    float phi = 2 * M_PI * r2;
    float x = r * cos(phi);
    float z = r * sin(phi);
    return Vec3(x, r1, z);
}

void create_orthonormal_system(Vec3 &n, Vec3 &rotX, Vec3 &rotY) {
    Vec3 closest_vec;
    if (n.x > n.y) {
        closest_vec = Vec3(n.z, 0, -n.x); // x-z plane 
    } else {
        closest_vec = Vec3(0, -n.z, n.y); // y-z plane
    }
    rotX = closest_vec.norm();
    rotY = n.cross(rotX);
}

void trace(Ray &ray, Vec3 &color, Scene &scene, int depth) {
    // Determine the color of certain pixel
    std::shared_ptr<SceneObj> obj;
    Intersection hit;

    if (depth >= scene.bounces) {
        color = scene.black;
        return;
    }

    bool intersect = get_intersection(ray, obj, hit, scene);
    if (!intersect) {
        color = scene.gray;
        return;
    }

    Material material = obj->material;
    Vec3 emission = obj->emission;

    Vec3 diffuse_color, reflective_color, refractive_color;
    if (material.type == DIFFUSE) {
        Vec3 direct_lighting = Vec3(0, 0, 0);

        for (int i=0; i < scene.scene_lights.size(); i++) {
            auto light = scene.scene_lights.at(i);
            Vec3 light_direction = light->direction(hit);
            double intensity = light->intensity(hit);

            // Shadow test
            Ray shadow_ray = Ray(hit.point + hit.normal * BIAS, light_direction);
            int visibility = calc_shadows(shadow_ray, light, hit, scene);

            // Lambert's law
            double cos_theta = light_direction.dot(hit.normal);
            if (cos_theta < 0) continue;

            Vec3 direct_color = light->color * (visibility * cos_theta * intensity);
            direct_lighting = direct_lighting + direct_color;
        }

        Vec3 indirect_lighting = Vec3(0, 0, 0);
        
        if (scene.global_illumination) {
            Vec3 rotX, rotY;
            create_orthonormal_system(hit.normal, rotX, rotY);
            float pdf = 1 / (2 * M_PI);

            // Trace indirect rays and accumulate results (Monte Carlo integration)
            for (int i=0; i < scene.secondary_rays; i++) {
                float r1 = random_number();
                float r2 = random_number();
                Vec3 random_direction = sample_hemisphere(r1, r2);
                // Transform samples to local coordinate system
                float sample_x = Vec3(rotY.x, hit.normal.x, rotX.x).dot(random_direction);
                float sample_y = Vec3(rotY.y, hit.normal.y, rotX.y).dot(random_direction);
                float sample_z = Vec3(rotY.z, hit.normal.z, rotX.z).dot(random_direction);
                Vec3 sample_direction = Vec3(sample_x, sample_y, sample_z);
                Ray diffuse_ray = Ray(hit.point, sample_direction);

                float cos_theta = sample_direction.dot(hit.normal);
                Vec3 trace_color;
                trace(diffuse_ray, trace_color, scene, depth+1);
                trace_color = trace_color * cos_theta;
                indirect_lighting = indirect_lighting + trace_color;
            }

            // Average by number of samples and divide by pdf of random variable (same for all rays)
            indirect_lighting = indirect_lighting / (scene.secondary_rays * pdf);
        } 

        // Apply the rendering equation
        Vec3 brdf = obj->color/M_PI; 
        Vec3 point_color = (direct_lighting + indirect_lighting).multVbyV(brdf);
        diffuse_color = emission + point_color;
    }
    else if (material.type == REFLECTIVE) {
        Vec3 I = ray.direction;
        Vec3 N = hit.normal;
        float cos_theta = N.dot(I);
        Vec3 reflected_direction = I - (N * (2 * cos_theta));
        Ray reflective_ray = Ray(hit.point, reflected_direction.norm());
        trace(reflective_ray, reflective_color, scene, depth+1);
    }
    else if (material.type == REFRACTIVE) {
        Vec3 I = ray.direction;
        Vec3 N = hit.normal;
        float eta = material.ior;

        float cos_theta = N.dot(I);
        // inside surface
        if (cos_theta > 0) { 
            N = N * -1;
            eta = 1 / eta;
        // outside surface
        } else {
            cos_theta = -cos_theta;    
        }
        eta = 1 / eta;

        float k = 1 - (eta * eta) * (1 - (cos_theta * cos_theta));
        Vec3 refractive_direction;
        if (k < 0) {
            refractive_direction = I - (N * (2 * cos_theta));
        } else {
            Vec3 eta_i = I * eta;
            float constant = (cos_theta * eta) + sqrt(k);
            refractive_direction = eta_i - (N * constant);
        }
        Ray refractive_ray = Ray(hit.point + N * -BIAS, refractive_direction.norm());
        trace(refractive_ray, refractive_color, scene, depth+1);
    }

    color = material.calc_final_color(diffuse_color, reflective_color, refractive_color); 
}

void render(Scene &scene) {
    for (int x = 0; x < scene.width; x++) {
        for (int y = 0; y < scene.height; y++) {
            Vec3 pixel = Vec3(0, 0, 0);
            Ray ray = scene.cam.get_ray(x, y, scene.width, scene.height);
            Vec3 color;
            trace(ray, color, scene, 0);
            pixel = pixel + color;
            write_color(scene.img, x, y, color, scene.gamma_val);
        }
    }
}

int main(int argc, char** argv) {
    // Read scene file
    auto reader = SceneReader();
    reader.process_scene_file(argv[1]);

    // Render scene
    render(reader.scene);

    // Save the image
    reader.scene.img.save_png(reader.scene.filename.c_str());

    return 0;
}
