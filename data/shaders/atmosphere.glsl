/* Stellarium Web Engine - Copyright (c) 2018 - Noctua Software Ltd
 *
 * This program is licensed under the terms of the GNU AGPL v3, or
 * alternatively under a commercial licence.
 *
 * The terms of the AGPL v3 license can be found in the main directory of this
 * repository.
 */

#ifdef GL_ES
precision mediump float;
#endif

uniform highp float u_atm_p[12];
uniform highp vec3  u_sun;
uniform highp float u_tm[3]; // Tonemapping koefs.

varying lowp    vec4        v_color;

#ifdef VERTEX_SHADER

attribute highp   vec4       a_pos;
attribute highp   vec3       a_sky_pos;
attribute highp   float      a_luminance;

highp float gammaf(highp float c)
{
    if (c < 0.0031308)
      return 19.92 * c;
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

vec3 xyy_to_srgb(highp vec3 xyy)
{
    const highp mat3 xyz_to_rgb = mat3(3.2406, -0.9689, 0.0557,
                                      -1.5372, 1.8758, -0.2040,
                                      -0.4986, 0.0415, 1.0570);
    highp vec3 xyz = vec3(xyy[0] * xyy[2] / xyy[1], xyy[2],
               (1.0 - xyy[0] - xyy[1]) * xyy[2] / xyy[1]);
    highp vec3 srgb = xyz_to_rgb * xyz;
    clamp(srgb, 0.0, 1.0);
    return srgb;
}

void main()
{
    highp vec3 xyy;
    highp float cos_gamma, cos_gamma2, gamma, cos_theta;
    highp vec3 p = a_sky_pos;

    gl_Position = a_pos;

    // First compute the xy color component (chromaticity) from Preetham model
    // and re-inject a_luminance for Y component (luminance).
    p[2] = abs(p[2]); // Mirror below horizon.
    cos_gamma = dot(p, u_sun);
    cos_gamma2 = cos_gamma * cos_gamma;
    gamma = acos(cos_gamma);
    cos_theta = p[2];
constexpr int SCATTERING_TEXTURE_R_SIZE = 32;
constexpr int SCATTERING_TEXTURE_MU_SIZE = 128;
constexpr int SCATTERING_TEXTURE_MU_S_SIZE = 32;
constexpr int SCATTERING_TEXTURE_MU_S_SIZE = 128;
constexpr int SCATTERING_TEXTURE_NU_SIZE = 8;

constexpr int SCATTERING_TEXTURE_WIDTH =
    SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
    SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_R_SIZE;
constexpr int SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
constexpr int SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;
constexpr int SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_MU_S_SIZE;

constexpr int IRRADIANCE_TEXTURE_WIDTH = 64;
constexpr int IRRADIANCE_TEXTURE_HEIGHT = 16;
constexpr int IRRADIANCE_TEXTURE_WIDTH = SCATTERING_TEXTURE_MU_SIZE;
constexpr int IRRADIANCE_TEXTURE_HEIGHT = SCATTERING_TEXTURE_R_SIZE;

// The conversion factor between watts and lumens.
constexpr double MAX_LUMINOUS_EFFICACY = 683.0;
 43  atmosphere/demo/demo.cc 
@@ -42,6 +42,7 @@ independent of our atmosphere model. The only part which is related to it is the
#include <glad/glad.h>
#include <GL/freeglut.h>

#include <cassert>
#include <algorithm>
#include <cmath>
#include <map>
@@ -77,7 +78,9 @@ const char kVertexShader[] = R"(
    uniform mat4 view_from_clip;
    layout(location = 0) in vec4 vertex;
    out vec3 view_ray;
    out vec3 position;
    void main() {
      position=vertex.xyz;
      view_ray =
          (model_from_view * vec4((view_from_clip * vertex).xyz, 0.0)).xyz;
      gl_Position = vertex;
@@ -103,9 +106,9 @@ Demo::Demo(int viewport_width, int viewport_height) :
    use_constant_solar_spectrum_(false),
    use_ozone_(true),
    use_combined_textures_(true),
    use_half_precision_(true),
    use_luminance_(NONE),
    do_white_balance_(false),
    use_half_precision_(false),
    use_luminance_(APPROXIMATE),
    do_white_balance_(true),
    show_help_(true),
    program_(0),
    view_distance_meters_(9000.0),
@@ -178,6 +181,7 @@ Demo::~Demo() {
  INSTANCES.erase(window_id_);
}

constexpr double kBottomRadius = 6360000.0;
/*
<p>The "real" initialization work, which is specific to our atmosphere model,
is done in the following method. It starts with the creation of an atmosphere
@@ -223,15 +227,14 @@ void Demo::InitModel() {
  // Wavelength independent solar irradiance "spectrum" (not physically
  // realistic, but was used in the original implementation).
  constexpr double kConstantSolarIrradiance = 1.5;
  constexpr double kBottomRadius = 6360000.0;
  constexpr double kTopRadius = 6420000.0;
  constexpr double kRayleigh = 1.24062e-6;
  constexpr double kRayleighScaleHeight = 8000.0;
  constexpr double kMieScaleHeight = 1200.0;
  constexpr double kMieAngstromAlpha = 0.0;
  constexpr double kMieAngstromBeta = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  constexpr double kMiePhaseFunctionG = 0.8;
  constexpr double kMiePhaseFunctionG = 0.76;
  constexpr double kGroundAlbedo = 0.1;
  const double max_sun_zenith_angle =
      (use_half_precision_ ? 102.0 : 120.0) / 180.0 * kPi;
@@ -294,6 +297,10 @@ to get the final scene rendering program:
  glShaderSource(vertex_shader_, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader_);

  GLint compile_status;
  glGetShaderiv(vertex_shader_, GL_COMPILE_STATUS, &compile_status);
  assert(compile_status == GL_TRUE);

  const std::string fragment_shader_str =
      "#version 330\n" +
      std::string(use_luminance_ != NONE ? "#define USE_LUMINANCE\n" : "") +
@@ -305,6 +312,9 @@ to get the final scene rendering program:
  glShaderSource(fragment_shader_, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader_);

  glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &compile_status);
  assert(compile_status == GL_TRUE);

  if (program_ != 0) {
    glDeleteProgram(program_);
  }
@@ -313,6 +323,12 @@ to get the final scene rendering program:
  glAttachShader(program_, fragment_shader_);
  glAttachShader(program_, model_->shader());
  glLinkProgram(program_);

  GLint link_status;
  glGetProgramiv(program_, GL_LINK_STATUS, &link_status);
  assert(link_status == GL_TRUE);
  assert(glGetError() == 0);

  glDetachShader(program_, vertex_shader_);
  glDetachShader(program_, fragment_shader_);
  glDetachShader(program_, model_->shader());
@@ -348,6 +364,14 @@ because our demo app does not have any texture of its own):
  HandleReshapeEvent(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
}

double getPointAltitude(double x, double y, double z)
{
    // Earth center coords
    const double c[3]={0.0, 0.0, -kBottomRadius / kLengthUnitInMeters};
    const auto distFromCenter=std::hypot(x-c[0],std::hypot(y-c[1],z-c[2]));
    return distFromCenter-kBottomRadius / kLengthUnitInMeters;
}

/*
<p>The scene rendering method simply sets the uniforms related to the camera
position and to the Sun direction, and then draws a full screen quad (and
@@ -411,6 +435,13 @@ void Demo::HandleRedisplayEvent() const {
         << (do_white_balance_ ? "on" : "off") << ")\n"
         << " +/-: increase/decrease exposure (" << exposure_ << ")\n"
         << " 1-9: predefined views\n";
    help << "\n"
         << "Sun elevation : " << 90-sun_zenith_angle_radians_*180/M_PI << " deg\n"
         << "Sun azimuth   : " << sun_azimuth_angle_radians_*180/M_PI << " deg\n"
         << "View elevation: " << view_zenith_angle_radians_*180/M_PI-90 << " deg\n"
         << "View azimuth  : " << view_azimuth_angle_radians_*180/M_PI << " deg\n"
         << "View distance : " << view_distance_meters_/1000 << " km\n"
         << "Camera altitude: " << getPointAltitude(model_from_view[3], model_from_view[7], model_from_view[11])*kLengthUnitInMeters/1000 << " km\n";
    text_renderer_->SetColor(1.0, 0.0, 0.0);
    text_renderer_->DrawText(help.str(), 5, 4);
  }
@@ -507,7 +538,7 @@ void Demo::HandleMouseClickEvent(
}

void Demo::HandleMouseDragEvent(int mouse_x, int mouse_y) {
  constexpr double kScale = 500.0;
  constexpr double kScale = 5000.0;
  if (is_ctrl_key_pressed_) {
    sun_zenith_angle_radians_ -= (previous_mouse_y_ - mouse_y) / kScale;
    sun_zenith_angle_radians_ =
 35  atmosphere/demo/demo.glsl 
@@ -59,7 +59,7 @@ luminance values (see <a href="../model.h.html">model.h</a>).

const float PI = 3.14159265;
const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;
const float kSphereRadius = 1000.0 / kLengthUnitInMeters;
const float kSphereRadius = 1.0 / kLengthUnitInMeters;
const vec3 kSphereAlbedo = vec3(0.8);
const vec3 kGroundAlbedo = vec3(0.0, 0.0, 0.04);

@@ -247,9 +247,40 @@ shadow volume of the sphere, because they are needed to get the aerial
perspective for the sphere and the planet:
*/

vec3 dither(vec3 c)
{
    const float bayerPattern[] = float[](
        0,  32,  8, 40,  2, 34, 10, 42,  /* 8x8 Bayer ordered dithering  */
        48, 16, 56, 24, 50, 18, 58, 26,  /* pattern.  Each input pixel   */
        12, 44,  4, 36, 14, 46,  6, 38,  /* is scaled to the 0..63 range */
        60, 28, 52, 20, 62, 30, 54, 22,  /* before looking in this table */
        3,  35, 11, 43,  1, 33,  9, 41,  /* to determine the action.     */
        51, 19, 59, 27, 49, 17, 57, 25,
        15, 47,  7, 39, 13, 45,  5, 37,
        63, 31, 55, 23, 61, 29, 53, 21);
    float bayer=bayerPattern[int(mod(int(gl_FragCoord.x),8)+8*mod(int(gl_FragCoord.y),8))]/64;
    vec3 rgb=c*255;
    vec3 head=floor(rgb);
    vec3 tail=fract(rgb);
    return (head+1-step(tail,vec3(bayer)))/255;
}

in vec3 position;
void main() {
  // Normalized view direction vector.
  vec3 view_direction = normalize(view_ray);
  {
      // Fisheye
      float x=position.x, y=position.y; // x,y\in[-1,1]
      float R=sqrt(x*x+y*y);
      float theta=2*asin(R);
      float phi=atan(y,x);
      view_direction=vec3(
                          cos(phi)*sin(theta),
                          sin(phi)*sin(theta),
                          1*cos(theta)
                         );
  }
  // Tangent of the angle subtended by this fragment.
  float fragment_angular_size =
      length(dFdx(view_ray) + dFdy(view_ray)) / length(view_ray);
@@ -382,6 +413,6 @@ the scene:
  radiance = mix(radiance, ground_radiance, ground_alpha);
  radiance = mix(radiance, sphere_radiance, sphere_alpha);
  color.rgb = 
      pow(vec3(1.0) - exp(-radiance / white_point * exposure), vec3(1.0 / 2.2));
      dither(pow(radiance / white_point * (exposure*exp(-50*dot(normalize(camera-earth_center),sun_direction))), vec3(1.0 / 2.2)));
  color.a = 1.0;
}
 55  atmosphere/demo/webgl/precompute.cc 
@@ -40,13 +40,16 @@ the OpenGL API, and are saved to disk:
#include <GL/freeglut.h>

#include <memory>
#include <cstring>
#include <fstream>

#include "atmosphere/demo/demo.h"
#include "atmosphere/constants.h"

using atmosphere::demo::Demo;

bool combinedMieAndRayleigh=false;

void SaveShader(const GLuint shader, const std::string& filename) {
  GLint sourceLength;
  glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &sourceLength);
@@ -55,20 +58,42 @@ void SaveShader(const GLuint shader, const std::string& filename) {
  std::unique_ptr<GLchar[]> buffer(new GLchar[sourceLength]);
  glGetShaderSource(shader, sourceLength, &actualLength, buffer.get());

  if(std::strstr(buffer.get(),"#define COMBINED_SCATTERING_TEXTURES\n"))
      combinedMieAndRayleigh=true;

  std::ofstream output_stream(filename, std::ofstream::out);
  output_stream << std::string(buffer.get());
  output_stream.close();
}

void SaveTexture(const GLenum texture_unit, const GLenum texture_target,
    const int texture_size, const std::string& filename) {
    const int texture_width, const int texture_height, const int texture_depth,
    const std::string& filename) {

  const int texture_size=texture_width*texture_height*texture_depth;
  std::unique_ptr<float[]> pixels(new float[texture_size * 4]);
  glActiveTexture(texture_unit);
  glGetTexImage(texture_target, 0, GL_RGBA, GL_FLOAT, pixels.get());

  std::ofstream output_stream(
      filename, std::ofstream::out | std::ofstream::binary);
  output_stream.write((const char*) pixels.get(), texture_size * 16);

  // The header goes to the end to avoid breaking the format for webgl demo
  if(texture_target==GL_TEXTURE_2D)
  {
      const std::int32_t header[]={texture_width, texture_height, 2};
      output_stream.write(reinterpret_cast<const char*>(header), sizeof header);
  }
  else // texture_target==GL_TEXTURE_3D
  {
      using namespace atmosphere;
      const std::int32_t header[]={SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE,
                                   SCATTERING_TEXTURE_R_SIZE, SCATTERING_TEXTURE_NU_SIZE,
                                   4};
      output_stream.write(reinterpret_cast<const char*>(header), sizeof header);
  }

  output_stream.close();
}

@@ -79,7 +104,7 @@ int main(int argc, char** argv) {
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  std::unique_ptr<Demo> demo(new Demo(0, 0));
  demo->model().SetProgramUniforms(demo->program(), 0, 1, 2);
  demo->model().SetProgramUniforms(demo->program(), 0, 1, 2, 3);

  const std::string output_dir(argv[1]);
  SaveShader(demo->model().shader(), output_dir + "atmosphere_shader.txt");
@@ -88,22 +113,34 @@ int main(int argc, char** argv) {
  SaveTexture(
      GL_TEXTURE0,
      GL_TEXTURE_2D,
      atmosphere::TRANSMITTANCE_TEXTURE_WIDTH *
          atmosphere::TRANSMITTANCE_TEXTURE_HEIGHT,
      atmosphere::TRANSMITTANCE_TEXTURE_WIDTH,
      atmosphere::TRANSMITTANCE_TEXTURE_HEIGHT,
      1,
      output_dir + "transmittance.dat");
  SaveTexture(
      GL_TEXTURE1,
      GL_TEXTURE_3D,
      atmosphere::SCATTERING_TEXTURE_WIDTH *
          atmosphere::SCATTERING_TEXTURE_HEIGHT *
          atmosphere::SCATTERING_TEXTURE_DEPTH,
      atmosphere::SCATTERING_TEXTURE_WIDTH,
      atmosphere::SCATTERING_TEXTURE_HEIGHT,
      atmosphere::SCATTERING_TEXTURE_DEPTH,
      output_dir + "scattering.dat");
  SaveTexture(
      GL_TEXTURE2,
      GL_TEXTURE_2D,
      atmosphere::IRRADIANCE_TEXTURE_WIDTH *
          atmosphere::IRRADIANCE_TEXTURE_HEIGHT,
      atmosphere::IRRADIANCE_TEXTURE_WIDTH,
      atmosphere::IRRADIANCE_TEXTURE_HEIGHT,
      1,
      output_dir + "irradiance.dat");
  if(!combinedMieAndRayleigh)
  {
      SaveTexture(
          GL_TEXTURE3,
          GL_TEXTURE_3D,
          atmosphere::SCATTERING_TEXTURE_WIDTH,
          atmosphere::SCATTERING_TEXTURE_HEIGHT,
          atmosphere::SCATTERING_TEXTURE_DEPTH,
          output_dir + "mie_scattering.dat");
  }

  return 0;
}
 38  atmosphere/functions.glsl 
@@ -743,7 +743,7 @@ InverseSolidAngle RayleighPhaseFunction(Number nu) {

InverseSolidAngle MiePhaseFunction(Number g, Number nu) {
  InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
  return (k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5) + 1./((1.-nu)*600.+0.05)/sr)*0.904;
}

/*
@@ -811,18 +811,11 @@ vec4 GetScatteringTextureUvwzFromRMuMuSNu(IN(AtmosphereParameters) atmosphere,
        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  Length d = DistanceToTopAtmosphereBoundary(
      atmosphere, atmosphere.bottom_radius, mu_s);
  Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
  Length d_max = H;
  Number a = (d - d_min) / (d_max - d_min);
  Number A =
      -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
  Number u_mu_s = GetTextureCoordFromUnitRange(
      max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);
      max(0,0.777411079203886 + 1.05536823610174*mu_s - SafeSqrt(0.00614636921559353 + (-0.130928575006726 + 0.818303593792038*mu_s)*mu_s)), SCATTERING_TEXTURE_MU_S_SIZE);

  Number u_nu = (nu + 1.0) / 2.0;
  return vec4(u_nu, u_mu_s, u_mu, u_r);
  return vec4(u_nu, u_r, u_mu, u_mu_s);
}

/*
@@ -842,7 +835,7 @@ void GetRMuMuSNuFromScatteringTextureUvwz(IN(AtmosphereParameters) atmosphere,
      atmosphere.bottom_radius * atmosphere.bottom_radius);
  // Distance to the horizon.
  Length rho =
      H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
      H * GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_R_SIZE);
  r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

  if (uvwz.z < 0.5) {
@@ -870,15 +863,8 @@ void GetRMuMuSNuFromScatteringTextureUvwz(IN(AtmosphereParameters) atmosphere,
  }

  Number x_mu_s =
      GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
  Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
  Length d_max = H;
  Number A =
      -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
  Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
  Length d = d_min + min(a, A) * (d_max - d_min);
  mu_s = d == 0.0 * m ? Number(1.0) :
     ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));
      GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_MU_S_SIZE);
  mu_s=ClampCosine(-2.9980496922533 + 3.57148399989121*x_mu_s + SafeSqrt(6.96385318200146 + x_mu_s*(-16.1532811495142 + 9.37138625741308*x_mu_s)));

  nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}
@@ -902,15 +888,15 @@ void GetRMuMuSNuFromScatteringTextureFragCoord(
    OUT(bool) ray_r_mu_intersects_ground) {
  const vec4 SCATTERING_TEXTURE_SIZE = vec4(
      SCATTERING_TEXTURE_NU_SIZE - 1,
      SCATTERING_TEXTURE_MU_S_SIZE,
      SCATTERING_TEXTURE_R_SIZE,
      SCATTERING_TEXTURE_MU_SIZE,
      SCATTERING_TEXTURE_R_SIZE);
      SCATTERING_TEXTURE_MU_S_SIZE);
  Number frag_coord_nu =
      floor(frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));
  Number frag_coord_mu_s =
      mod(frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));
      floor(frag_coord.x / Number(SCATTERING_TEXTURE_R_SIZE));
  Number frag_coord_r =
      mod(frag_coord.x, Number(SCATTERING_TEXTURE_R_SIZE));
  vec4 uvwz =
      vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
      vec4(frag_coord_nu, frag_coord_r, frag_coord.y, frag_coord.z) /
          SCATTERING_TEXTURE_SIZE;
  GetRMuMuSNuFromScatteringTextureUvwz(
      atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);1056,6 +1056,8 @@ void Model::Precompute(
    const mat3& luminance_from_radiance,
    bool blend,
    unsigned int num_scattering_orders) {

  std::cerr << "Precompute(lambdas={" << lambdas[0] << ',' << lambdas[1] << ',' << lambdas[2] << "})\n";
  // The precomputations require specific GLSL programs, for each precomputation
  // step. We create and compile them here (they are automatically destroyed
  // when this method returns, via the Program destructor).
@@ -1083,13 +1085,17 @@ void Model::Precompute(
  glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

  // Compute the transmittance, and store it in transmittance_texture_.
  std::cerr << " Computing transmittance... ";
  glFramebufferTexture(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, transmittance_texture_, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glViewport(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  compute_transmittance.Use();
  DrawQuad({}, full_screen_quad_vao_);
  glFinish();
  std::cerr << "done\n";

  std::cerr << " Computing direct irradiance... ";
  // Compute the direct irradiance, store it in delta_irradiance_texture and,
  // depending on 'blend', either initialize irradiance_texture_ with zeros or
  // leave it unchanged (we don't want the direct irradiance in
@@ -1104,6 +1110,8 @@ void Model::Precompute(
  compute_direct_irradiance.BindTexture2d(
      "transmittance_texture", transmittance_texture_, 0);
  DrawQuad({false, blend}, full_screen_quad_vao_);
  glFinish();
  std::cerr << "done\n";

  // Compute the rayleigh and mie single scattering, store them in
  // delta_rayleigh_scattering_texture and delta_mie_scattering_texture, and
@@ -1128,15 +1136,22 @@ void Model::Precompute(
      "luminance_from_radiance", luminance_from_radiance);
  compute_single_scattering.BindTexture2d(
      "transmittance_texture", transmittance_texture_, 0);
  std::cerr << " Computing single scattering layers... ";
  for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
    std::cerr << layer;
    compute_single_scattering.BindInt("layer", layer);
    DrawQuad({false, false, blend, blend}, full_screen_quad_vao_);
    glFinish();
    if(layer+1<SCATTERING_TEXTURE_DEPTH) std::cerr << ",";
  }
  std::cerr << "\n";

  // Compute the 2nd, 3rd and 4th order of scattering, in sequence.
  std::cerr << " Computing multiple scattering...\n";
  for (unsigned int scattering_order = 2;
       scattering_order <= num_scattering_orders;
       ++scattering_order) {
    std::cerr << "  Scattering order " << scattering_order << "\n";
    // Compute the scattering density, and store it in
    // delta_scattering_density_texture.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
@@ -1160,11 +1175,17 @@ void Model::Precompute(
    compute_scattering_density.BindTexture2d(
        "irradiance_texture", delta_irradiance_texture, 4);
    compute_scattering_density.BindInt("scattering_order", scattering_order);
    std::cerr << "   Computing scattering density layers... ";
    for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
      std::cerr << layer;
      compute_scattering_density.BindInt("layer", layer);
      DrawQuad({}, full_screen_quad_vao_);
      glFinish();
      if(layer+1<SCATTERING_TEXTURE_DEPTH) std::cerr << ",";
    }
    std::cerr << "\n";

    std::cerr << "   Computing indirect irradiance... ";
    // Compute the indirect irradiance, store it in delta_irradiance_texture and
    // accumulate it in irradiance_texture_.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
@@ -1186,7 +1207,9 @@ void Model::Precompute(
        "multiple_scattering_texture", delta_multiple_scattering_texture, 2);
    compute_indirect_irradiance.BindInt("scattering_order",
        scattering_order - 1);
    glFinish();
    DrawQuad({false, true}, full_screen_quad_vao_);
    std::cerr << "done\n";

    // Compute the multiple scattering, store it in
    // delta_multiple_scattering_texture, and accumulate it in
@@ -1204,11 +1227,17 @@ void Model::Precompute(
        "transmittance_texture", transmittance_texture_, 0);
    compute_multiple_scattering.BindTexture3d(
        "scattering_density_texture", delta_scattering_density_texture, 1);
    std::cerr << "   Computing multiple scattering layers... ";
    for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
      std::cerr << layer;
      compute_multiple_scattering.BindInt("layer", layer);
      DrawQuad({false, true}, full_screen_quad_vao_);
      glFinish();
      if(layer+1<SCATTERING_TEXTURE_DEPTH) std::cerr << ",";
    }
    std::cerr << "\n";
  }
  std::cerr << "Model::Precompute() done\n";
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);

#endif
#ifdef FRAGMENT_SHADER

void main()
{
    gl_FragColor = v_color;
}

#endif
