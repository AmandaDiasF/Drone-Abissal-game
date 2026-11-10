#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABISSAL-1: O  RESGATE SUBMARINO  --  minijogo 3D (Python + PyOpenGL + GLFW, OpenGL 3.3 core)

PIPELINE GRAFICA (o que acontece com cada vertice, na ordem):
  1. Coordenadas LOCAIS : vertices do modelo (ex.: cubo unitario de -0.5 a 0.5).
  2. MUNDO  : p_mundo = Model * p_local, com Model = T * R * S (escala, depois rotacao, depois translacao).
              Para objetos hierarquicos (drone, peixes, corais), Model = Modelo_do_pai * Modelo_local_da_parte.
  3. CAMERA : p_camera = View * p_mundo  (matriz look_at: coloca a camera na origem olhando para -Z).
  4. PROJECAO: p_clip = Proj * p_camera  (perspectiva: objetos distantes ficam menores).
  5. A GPU divide por w (NDC), faz recorte, viewport e TESTE DE PROFUNDIDADE (z-buffer),
     que garante a oclusao correta (drone some atras de um coral).
  No codigo: gl_Position = uProj * uView * uModel * vec4(aPos, 1.0)  (vertex shader).

ILUMINACAO: a luz principal e uma LANTERNA (spot light) presa na frente do drone. A intensidade depende do
angulo entre o eixo da lanterna e a direcao do fragmento (smoothstep entre cone interno e externo): no centro do
cone a luz e forte e vai diminuindo ate zero conforme o drone "mira" para outro lugar. Tecla L liga/desliga.

Convencao de matrizes: 4x4 numpy, vetores-coluna (M @ v), enviadas ao shader com transpose=GL_TRUE.
"""
import math
import sys
import ctypes
import os
import struct
import time
import zlib
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

TITLE = "Abissal-1: O Resgate Submarino"

# ----------------------------------------------------------------------------
# PARAMETROS DO CENARIO
# ----------------------------------------------------------------------------
HALF = 15.0                       # arena: quadrado de lado 2*HALF
DRONE_R = 0.7                     # raio de colisao do drone
DRONE_H = 1.0                     # altura (aprox. constante)
SPEED = 6.0                       # unidades/s
COLLECT_TIME = 0.35               # duracao da animacao de coleta (s)
START = (-11.0, 11.0)
PORTAL = (11.0, -11.0)
CELLS = [(-8, 6), (1, 7), (8, 3), (7, -6), (-7, -2)]                      # 5 celulas de energia
CORALS = [(-3, 3, 1.2, 0, (1.0, 0.35, 0.50)),                             # obstaculos fixos: x, z, raio, variacao, cor
          (4, 0, 1.3, 1, (0.65, 0.35, 0.95)),
          (-2, -6, 1.3, 2, (1.0, 0.45, 0.75))]
FISH = [dict(cx=-7, cz=0, R=3.0, w=0.9, ph=0.0, col=(0.95, 0.35, 0.55)),  # obstaculos moveis: orbitam um ponto
        dict(cx=7, cz=7, R=2.5, w=-1.0, ph=2.0, col=(0.55, 0.40, 0.95)),
        dict(cx=1, cz=-3, R=2.3, w=0.8, ph=4.0, col=(0.20, 0.85, 0.70))]
FISH_R = 0.55                                                             # raio de colisao dos peixes
# ---- PARAMETROS AJUSTAVEIS (faceis de alterar durante a apresentacao) ----
LAMP_OFFSET = (0.0, -0.07, 0.75)  # posicao da lanterna no referencial do drone (x, y, z); ex.: y = 3 -> luz vinda de cima
LAMP_TILT = math.radians(8)       # inclinacao da lanterna para baixo
CONE_IN_DEG, CONE_OUT_DEG = 10.0, 38.0   # cone: intensidade plena ate IN, zero a partir de OUT
LIGHT_INTENSITY = 3.0             # intensidade da lanterna
CELL_SPIN_DEG = 90.0              # velocidade de rotacao das celulas (graus/s)
CELL_BOB_AMP, CELL_BOB_FREQ = 0.25, 2.0  # amplitude e frequencia da flutuacao das celulas
CELL_SCALE = 1.0                  # tamanho das celulas
DRONE_SCALE = 1.0                 # tamanho do drone
COS_IN, COS_OUT = math.cos(math.radians(CONE_IN_DEG)), math.cos(math.radians(CONE_OUT_DEG))

# ----------------------------------------------------------------------------
# SHADERS  (Blinn-Phong por fragmento: ambiente + lanterna difusa/especular, atenuacao e neblina)
# ----------------------------------------------------------------------------
VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uModel, uView, uProj;
uniform mat3 uNormalMat;               // inversa transposta de Model: mantem normais corretas com escala
out vec3 vWorld; out vec3 vNormal; out vec2 vUV;
void main(){
    vec4 w = uModel * vec4(aPos, 1.0);          // local -> mundo
    vWorld = w.xyz;
    vNormal = uNormalMat * aNormal;
    vUV = aUV;
    gl_Position = uProj * uView * w;            // mundo -> camera -> projecao
}
"""

FS = """
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform int   uTexMode;      // 0 = sem textura, 1 = usa UV do vertice (piso), 2 = projecao por caixa (paredes)
uniform float uTile;         // repeticoes por unidade de mundo (modo 2)
uniform vec3  uColor, uEmissive;
uniform float uShin, uSpec;
uniform float uAmb;          // multiplicador da luz ambiente (maior no drone/peixes para ficarem visiveis)
uniform float uFill;         // luz de cortesia (so no drone): realce especular no casco
uniform int   uUnlit, uLightOn;
uniform vec3  uLightPos, uLightDir, uLightColor, uCamPos, uFogColor;
uniform float uCosIn, uCosOut, uFogDensity;
void main(){
    vec3 base = uColor;
    if (uTexMode == 1) base *= texture(uTex, vUV).rgb;
    else if (uTexMode == 2) {
        // UV derivado da posicao de mundo pelo eixo dominante da normal: sem deformacao em caixas escaladas
        vec3 n = abs(normalize(vNormal)); vec2 uv;
        if (n.y >= n.x && n.y >= n.z) uv = vWorld.xz; else if (n.x >= n.z) uv = vWorld.zy; else uv = vWorld.xy;
        base *= texture(uTex, uv * uTile).rgb;
    }
    vec3 color;
    if (uUnlit == 1) color = base + uEmissive;
    else {
        vec3 N = normalize(vNormal), V = normalize(uCamPos - vWorld);
        vec3 ambient = base * vec3(0.18, 0.30, 0.44) * uAmb;          // luz ambiente azulada (agua), bem fraca
        vec3 lit = vec3(0.0);
        if (uLightOn == 1) {
            vec3 L = uLightPos - vWorld; float d = length(L); L /= d;
            float spot = smoothstep(uCosOut, uCosIn, dot(-L, normalize(uLightDir)));   // 1 no eixo, 0 fora do cone
            float att  = 1.0 / (1.0 + 0.08 * d + 0.012 * d * d);
            float diff = max(dot(N, L), 0.0);
            vec3  H    = normalize(L + V);
            float sp   = (diff > 0.0) ? pow(max(dot(N, H), 0.0), uShin) * uSpec : 0.0;
            lit = (base * diff + vec3(sp)) * uLightColor * att * spot;
        }
        if (uFill > 0.0 && uLightOn == 1) {   // luz de apoio do casco: tambem so existe com a luz ligada
            vec3 Lf = normalize(vec3(0.4, 1.0, 0.3)); float df = max(dot(N, Lf), 0.0);
            float sf = (df > 0.0) ? pow(max(dot(N, normalize(Lf + V)), 0.0), uShin) * uSpec : 0.0;
            lit += (base * df * 0.6 + vec3(sf) * 0.8) * uFill;
        }
        color = ambient + lit + uEmissive;
    }
    float f = clamp(exp(-uFogDensity * length(uCamPos - vWorld)), 0.0, 1.0);   // neblina submarina
    FragColor = vec4(mix(uFogColor, color, f), 1.0);
}
"""

# ----------------------------------------------------------------------------
# MATEMATICA: matrizes de transformacao
# ----------------------------------------------------------------------------
def T(x, y, z):                       # TRANSLACAO
    m = np.eye(4); m[:3, 3] = (x, y, z); return m

def S(x, y, z):                       # ESCALA
    return np.diag([x, y, z, 1.0])

def Ry(a):                            # ROTACAO em torno do eixo Y
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m

def Rz(a):                            # ROTACAO em torno do eixo Z (helices giram em torno do eixo de proa)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c
    return m

def perspective(fov, aspect, near, far):          # PROJECAO em perspectiva
    f = 1.0 / math.tan(fov / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far); m[2, 3] = 2 * far * near / (near - far); m[3, 2] = -1
    return m

def look_at(eye, ctr, up):                        # matriz de CAMERA (view)
    f = ctr - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4); m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m

# ----------------------------------------------------------------------------
# GEOMETRIA (vertice = pos3 + normal3 + uv2)
# ----------------------------------------------------------------------------
class Mesh:
    def __init__(self, data, mode):
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, 8)
        self.count, self.mode = len(data), mode
        self.vao = glGenVertexArrays(1); glBindVertexArray(self.vao)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        for i, (n, off) in enumerate(((3, 0), (3, 12), (2, 24))):
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(i, n, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(off))

    def draw(self):
        glBindVertexArray(self.vao); glDrawArrays(self.mode, 0, self.count)

def make_cube():                      # PRIMITIVA: GL_TRIANGLES (6 faces quadrilateras = 12 triangulos)
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
             ((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]   # (normal, u, v) com u x v = n
    v = []
    for n, u, w in faces:
        n, u, w = np.array(n), np.array(u), np.array(w)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)):
            p = n * 0.5 + u * a * 0.5 + w * b * 0.5
            v.append([*p, *n, (a + 1) / 2, (b + 1) / 2])
    return v

def make_octa():                      # octaedro (celulas e nadadeiras), normais planas por face
    v = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                a, b, c = np.array([sx, 0, 0.]), np.array([0, sy, 0.]), np.array([0, 0, sz * 1.])
                n = np.cross(b - a, c - a)
                if np.dot(n, a + b + c) < 0: b, c, n = c, b, -n
                n = n / np.linalg.norm(n)
                for p, uv in ((a, (0, 0)), (b, (1, 0)), (c, (.5, 1))): v.append([*p, *n, *uv])
    return v

def make_sphere(st=12, sl=20):        # esfera unitaria (casco do drone, corpo dos peixes, polipos)
    def P(i, j):
        th, ph = math.pi * i / st, 2 * math.pi * j / sl
        n = (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
        return [*n, *n, j / sl, i / st]
    v = []
    for i in range(st):
        for j in range(sl):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            v += [a, b, c, a, c, d]
    return v

def make_cone(n=14):                  # cone (galhos de coral): base raio 0.5 em y=0, ponta em y=1
    m = math.sqrt(1.25)
    nrm = lambda a: (math.cos(a) / m, 0.5 / m, math.sin(a) / m)
    v = []
    for j in range(n):
        a0, a1 = 2 * math.pi * j / n, 2 * math.pi * (j + 1) / n
        v += [[0.5 * math.cos(a0), 0, 0.5 * math.sin(a0), *nrm(a0), 0, 0],
              [0.5 * math.cos(a1), 0, 0.5 * math.sin(a1), *nrm(a1), 1, 0],
              [0, 1, 0, *nrm((a0 + a1) / 2), .5, 1]]
    return v

def make_torus(R=0.5, r=0.08, nM=24, nm=8):   # anel no plano XY (carenagem dos propulsores laterais)
    def P(i, j):
        t, p = 2 * math.pi * i / nM, 2 * math.pi * j / nm
        cp, sp = math.cos(p), math.sin(p)
        return [(R + r * cp) * math.cos(t), (R + r * cp) * math.sin(t), r * sp,
                cp * math.cos(t), cp * math.sin(t), sp, i / nM, j / nm]
    v = []
    for i in range(nM):
        for j in range(nm):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            v += [a, b, c, a, c, d]
    return v

def make_floor(half, rep):            # PRIMITIVA: GL_TRIANGLE_STRIP (quadrilatero), UV repetido 'rep' vezes
    up = (0, 1, 0)
    return [[-half, 0, -half, *up, 0, 0], [-half, 0, half, *up, 0, rep],
            [half, 0, -half, *up, rep, 0], [half, 0, half, *up, rep, rep]]

def make_grid(half, step):            # PRIMITIVA: GL_LINES (grade luminosa no piso)
    v, up = [], (0, 1, 0)
    for k in np.arange(-half, half + 0.01, step):
        v += [[k, 0.02, -half, *up, 0, 0], [k, 0.02, half, *up, 0, 0],
              [-half, 0.02, k, *up, 0, 0], [half, 0.02, k, *up, 0, 0]]
    return v

def make_circle(n=48):                # GL_LINE_LOOP no plano XZ (anel do portal)
    return [[math.cos(2 * math.pi * i / n), 0, math.sin(2 * math.pi * i / n), 0, 1, 0, 0, 0] for i in range(n)]

# ----------------------------------------------------------------------------
# TEXTURAS PROCEDURAIS (geradas por codigo, sem arquivos externos)
# ----------------------------------------------------------------------------
def make_floor_tex(n=256):            # textura 1: piso metalico/oceanico em placas
    rng = np.random.default_rng(1)
    img = np.ones((n, n, 3)) * np.array([0.20, 0.32, 0.40]) + rng.random((n, n, 1)) * 0.15
    t = n // 4
    for i in range(4):
        for j in range(4):
            img[i * t:(i + 1) * t, j * t:(j + 1) * t] *= 0.85 + 0.3 * rng.random()
    for k in range(0, n, t):
        img[k:k + 3, :] *= 0.35; img[:, k:k + 3] *= 0.35          # frestas entre placas
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def make_wall_tex(n=256):             # textura 2: painel de parede com faixa ciano e rebites
    rng = np.random.default_rng(2)
    img = np.ones((n, n, 3)) * np.array([0.42, 0.46, 0.52]) + rng.random((n, n, 1)) * 0.08
    yy, xx = np.mgrid[0:n, 0:n]
    h = n // 2
    lx, ly = xx % h, yy % h
    img[(lx < 3) | (ly < 3)] *= 0.35
    img[(ly > h // 2 - 4) & (ly < h // 2 + 4)] = [0.10, 0.55, 0.65]
    for cx in (10, h - 10):
        for cy in (10, h - 10):
            img[((lx - cx) ** 2 + (ly - cy) ** 2) < 16] = [0.80, 0.82, 0.85]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def upload_texture(img, unit):
    tid = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tid)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    h, w, _ = img.shape
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, np.ascontiguousarray(img))
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)       # repeticao das coordenadas de textura
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# ----------------------------------------------------------------------------
# RENDERER: encapsula shader e uniforms
# ----------------------------------------------------------------------------
class Renderer:
    def __init__(self):
        self.prog = compileProgram(compileShader(VS, GL_VERTEX_SHADER),
                                   compileShader(FS, GL_FRAGMENT_SHADER), validate=False)
        glUseProgram(self.prog); self.loc = {}

    def u(self, name):
        if name not in self.loc: self.loc[name] = glGetUniformLocation(self.prog, name)
        return self.loc[name]

    def frame(self, view, proj, eye, light_on, lamp_pos, lamp_dir):
        glUniformMatrix4fv(self.u("uView"), 1, GL_TRUE, view.astype(np.float32))
        glUniformMatrix4fv(self.u("uProj"), 1, GL_TRUE, proj.astype(np.float32))
        glUniform3f(self.u("uCamPos"), *map(float, eye))
        glUniform3f(self.u("uLightPos"), *map(float, lamp_pos))
        glUniform3f(self.u("uLightDir"), *map(float, lamp_dir))
        glUniform1f(self.u("uCosIn"), COS_IN); glUniform1f(self.u("uCosOut"), COS_OUT)
        glUniform3f(self.u("uLightColor"), *[c * LIGHT_INTENSITY for c in (0.90, 1.0, 1.0)])
        glUniform1i(self.u("uLightOn"), int(light_on))
        glUniform3f(self.u("uFogColor"), 0.01, 0.05, 0.10)
        glUniform1f(self.u("uFogDensity"), 0.025)

    def draw(self, mesh, M, color, tex=None, emis=(0, 0, 0), shin=16.0, spec=0.0, unlit=False, amb=1.0, fill=0.0):
        nm = np.linalg.inv(M[:3, :3]).T
        glUniformMatrix4fv(self.u("uModel"), 1, GL_TRUE, M.astype(np.float32))
        glUniformMatrix3fv(self.u("uNormalMat"), 1, GL_TRUE, nm.astype(np.float32))
        glUniform3f(self.u("uColor"), *map(float, color))
        glUniform3f(self.u("uEmissive"), *map(float, emis))
        glUniform1f(self.u("uShin"), shin); glUniform1f(self.u("uSpec"), spec)
        glUniform1f(self.u("uAmb"), amb); glUniform1f(self.u("uFill"), fill)
        glUniform1i(self.u("uUnlit"), int(unlit))
        glUniform1i(self.u("uTexMode"), {None: 0, "floor": 1, "wall": 2}[tex])
        glUniform1i(self.u("uTex"), 0 if tex == "floor" else 1)
        glUniform1f(self.u("uTile"), 0.3)
        mesh.draw()

# ----------------------------------------------------------------------------
# ESTADO DO JOGO (logica + animacoes, tudo escalado por dt)
# ----------------------------------------------------------------------------
def fish_pos(f):                      # posicao (x, z) e direcao do peixe na orbita
    a = f["a"]
    return f["cx"] + f["R"] * math.cos(a), f["cz"] + f["R"] * math.sin(a)

class Game:
    def __init__(self):
        self.light = True
        self.light_draw = True            # luz usada no desenho do quadro (muda no modo comparacao)
        self.reset()

    def reset(self):
        self.t = 0.0                                  # tempo acumulado (soma de dt)
        self.pos = np.array(START, dtype=float)       # posicao (x, z) do drone
        self.yaw = 3 * math.pi / 4                    # orientacao do drone (a lanterna aponta para onde ele olha)
        self.prop, self.prop_speed = 0.0, 1200.0      # angulo e velocidade (graus/s) das helices
        self.spin = 0.0                               # angulo de rotacao das celulas
        self.fish = [dict(f, a=f["ph"]) for f in FISH]
        self.cells = [dict(x=x, z=z, ph=i * 1.3, state=0, timer=0.0, y0=1.2) for i, (x, z) in enumerate(CELLS)]
        self.collected, self.portal_t, self.won, self.hit = 0, 0.0, False, 0.0

    def lamp(self):                                   # posicao e direcao da lanterna (na proa do drone)
        y = DRONE_H + 0.06 * math.sin(self.t * 3)
        pos = np.array([self.pos[0], y, self.pos[1]]) + Ry(self.yaw)[:3, :3] @ np.array(LAMP_OFFSET)   # offset no referencial do drone
        d = np.array([math.sin(self.yaw) * math.cos(LAMP_TILT), -math.sin(LAMP_TILT), math.cos(self.yaw) * math.cos(LAMP_TILT)])
        return pos, d

    def move(self, dt, win, cam_yaw):
        k = lambda key: glfw.get_key(win, key) == glfw.PRESS
        mx = k(glfw.KEY_D) - k(glfw.KEY_A); mz = k(glfw.KEY_W) - k(glfw.KEY_S)
        moving = bool(mx or mz)
        if moving:
            fwd = np.array([-math.sin(cam_yaw), -math.cos(cam_yaw)])      # frente relativa a camera (plano XZ)
            right = np.array([math.cos(cam_yaw), -math.sin(cam_yaw)])
            d = fwd * mz + right * mx; d /= np.linalg.norm(d)
            self.pos += d * SPEED * dt                                      # TRANSLACAO
            target = math.atan2(d[0], d[1])                                 # ROTACAO: vira para a direcao do movimento
            diff = (target - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += diff * min(1.0, 10 * dt)
        self.prop_speed += ((2800 if moving else 1200) - self.prop_speed) * min(1.0, 5 * dt)

    def push(self, cx, cz, r):                        # colisao circulo x circulo: empurra o drone para fora
        dx, dz = self.pos[0] - cx, self.pos[1] - cz
        d = math.hypot(dx, dz)
        if d < DRONE_R + r:
            if d < 1e-6: dx, dz, d = 1.0, 0.0, 1.0
            self.pos = np.array([cx + dx / d * (DRONE_R + r), cz + dz / d * (DRONE_R + r)])
            self.hit = 0.3                                                  # casco pisca em vermelho

    def collide(self):
        lim = HALF - DRONE_R
        self.pos = np.clip(self.pos, -lim, lim)                             # paredes da arena
        for cx, cz, rad, _, _ in CORALS: self.push(cx, cz, rad)            # corais (fixos)
        for f in self.fish: self.push(*fish_pos(f), FISH_R)                 # peixes (moveis)
        self.pos = np.clip(self.pos, -lim, lim)

    def update(self, dt, win, cam_yaw):
        self.t += dt
        self.spin += math.radians(CELL_SPIN_DEG) * dt                                  # rotacao continua das celulas
        self.hit = max(0.0, self.hit - dt)
        for f in self.fish: f["a"] += f["w"] * dt                           # peixes nadam em circulos
        if not self.won: self.move(dt, win, cam_yaw)
        self.collide()
        self.prop = (self.prop + math.radians(self.prop_speed) * dt) % (2 * math.pi)
        for c in self.cells:
            if c["state"] == 0 and math.hypot(c["x"] - self.pos[0], c["z"] - self.pos[1]) < DRONE_R + 0.45:
                c["state"], c["timer"], self.collected = 1, 0.0, self.collected + 1
                c["y0"] = 1.2 + CELL_BOB_AMP * math.sin(CELL_BOB_FREQ * self.t + c["ph"])
            elif c["state"] == 1:
                c["timer"] += dt
                if c["timer"] >= COLLECT_TIME: c["state"] = 2
        if self.collected == len(self.cells):
            self.portal_t = min(1.0, self.portal_t + dt / 1.2)              # ativacao animada do portal (1,2 s)
            if math.hypot(PORTAL[0] - self.pos[0], PORTAL[1] - self.pos[1]) < 1.3: self.won = True

# ----------------------------------------------------------------------------
# DESENHO DA CENA
# ----------------------------------------------------------------------------
def lerp(a, b, k): return tuple(x + (y - x) * k for x, y in zip(a, b))

def draw_drone(r, g, M):
    """Drone submarino (ROV). Hierarquia: casco (raiz) -> bracos -> carenagem -> helices; helice de popa; sensor frontal.
    Frente do drone = +Z local."""
    cube, sph, torus = M["cube"], M["sph"], M["torus"]
    orange, yellow, steel = (1.0, 0.55, 0.08), (1.0, 0.85, 0.15), (0.35, 0.38, 0.42)
    D = dict(amb=2.2, fill=1.0)                                             # drone sempre legivel, mesmo fora do facho
    flash = (0.7 * min(1.0, g.hit * 4), 0, 0)
    lamp_e = (1.0, 0.95, 0.7) if g.light_draw else (0.10, 0.10, 0.08)
    y = DRONE_H + 0.06 * math.sin(g.t * 3)                                  # leve flutuacao
    base = T(g.pos[0], y, g.pos[1]) @ Ry(g.yaw) @ S(DRONE_SCALE, DRONE_SCALE, DRONE_SCALE)                             # RAIZ: posicao + orientacao globais
    r.draw(sph, base @ S(0.40, 0.32, 0.80), orange, emis=flash, shin=64, spec=1.2, **D)            # casco ovoide
    r.draw(cube, base @ T(0, 0.32, -0.05) @ S(0.20, 0.20, 0.40), yellow, shin=64, spec=1.0, **D)   # torre dorsal
    r.draw(sph, base @ T(0, 0.02, 0.62) @ S(0.22, 0.20, 0.22), (0.1, 0.5, 0.6), emis=(0.2, 0.9, 1.0), **D)  # SENSOR frontal (domo)
    for s in (-1, 1):
        r.draw(sph, base @ T(s * 0.19, -0.07, 0.66) @ S(0.09, 0.09, 0.09), (1, 1, 0.9), emis=lamp_e)       # farois
    r.draw(cube, base @ T(0, 0.20, -0.68) @ S(0.04, 0.40, 0.30), steel, shin=80, spec=1.0, **D)     # leme vertical
    r.draw(cube, base @ T(0, 0.0, -0.66) @ S(0.80, 0.04, 0.25), steel, shin=80, spec=1.0, **D)      # estabilizador horizontal
    stern = base @ T(0, 0, -0.84)                                           # helice de popa (filha do casco)
    r.draw(sph, stern @ S(0.09, 0.09, 0.12), steel, shin=80, spec=1.0, **D)
    for k in range(3):
        r.draw(cube, stern @ Rz(g.prop * 1.5 + k * 2 * math.pi / 3) @ T(0.22, 0, 0) @ S(0.40, 0.07, 0.03), yellow, **D)
    for s in (-1, 1):                                                       # BRACOS laterais com propulsores
        arm = base @ T(s * 0.30, -0.02, 0)                                  # filho do casco
        r.draw(cube, arm @ T(s * 0.32, 0, 0) @ S(0.65, 0.07, 0.10), steel, shin=80, spec=1.0, **D)
        duct = arm @ T(s * 0.65, 0, 0)                                      # filho do braco
        r.draw(torus, duct @ S(0.7, 0.7, 0.7), steel, shin=80, spec=1.2, **D)    # anel (carenagem)
        r.draw(sph, duct @ S(0.06, 0.06, 0.06), steel, **D)
        for k in range(2):                                                  # helices: giram no proprio centro e acompanham o drone
            r.draw(cube, duct @ Rz(s * g.prop + k * math.pi / 2) @ S(0.55, 0.05, 0.035), yellow, **D)

def draw_coral(r, M, cx, cz, rad, var, col):
    base = T(cx, 0, cz)
    dim = tuple(0.18 * c for c in col)                                      # brilho fraco: coral visivel mesmo no escuro
    r.draw(M["octa"], base @ T(0, 0.15, 0) @ S(rad * 0.95, 0.5, rad * 0.95), (0.30, 0.28, 0.35), shin=8, spec=0.1)
    r.draw(M["cone"], base @ T(0, 0.3, 0) @ S(0.6, 2.6, 0.6), col, emis=dim, shin=16, spec=0.3)
    for i in range(6):
        k = (i + var) % 3
        arm = base @ T(0, 0.3, 0) @ Ry(math.radians(60 * i + 25 * var)) @ Rz(math.radians(24 + 8 * k))
        h = 1.5 + 0.4 * k
        r.draw(M["cone"], arm @ S(0.5, h, 0.5), col, emis=dim, shin=16, spec=0.3)
        r.draw(M["sph"], arm @ T(0, h, 0) @ S(0.13, 0.13, 0.13), (1.0, 0.9, 0.7), emis=(0.5, 0.4, 0.3))   # polipo luminoso

def draw_fish(r, g, M, f):
    x, z = fish_pos(f)
    sg = 1 if f["w"] > 0 else -1
    yaw = math.atan2(-math.sin(f["a"]) * sg, math.cos(f["a"]) * sg)         # olha para a direcao do nado
    y = 1.1 + 0.15 * math.sin(g.t * 2 + f["ph"])
    base = T(x, y, z) @ Ry(yaw)
    col = f["col"]; kw = dict(amb=2.0, shin=48, spec=0.8)
    glow = tuple(0.15 * c for c in col)
    r.draw(M["sph"], base @ S(0.22, 0.30, 0.55), col, emis=glow, **kw)                                  # corpo
    wag = 0.5 * math.sin(g.t * 8 + f["ph"])                                                             # cauda balanca (movimento periodico)
    r.draw(M["octa"], base @ T(0, 0, -0.5) @ Ry(wag) @ T(0, 0, -0.2) @ S(0.03, 0.30, 0.30), col, emis=glow, **kw)
    r.draw(M["octa"], base @ T(0, 0.28, -0.05) @ S(0.03, 0.20, 0.28), col, emis=glow, **kw)             # nadadeira dorsal
    for s in (-1, 1):
        r.draw(M["sph"], base @ T(s * 0.14, 0.08, 0.38) @ S(0.06, 0.06, 0.06), (1, 1, 1), emis=(0.6, 0.6, 0.6))

def draw_portal(r, g, M):
    k = g.portal_t * g.portal_t * (3 - 2 * g.portal_t)                      # suavizacao (smoothstep)
    pulse = 0.8 + 0.2 * math.sin(g.t * 6) if k >= 1 else 1.0
    col = lerp((0.10, 0.18, 0.15), (0.10, 1.00, 0.35), k)
    emis = tuple(c * 0.9 * k * pulse for c in col)
    p = T(PORTAL[0], 0, PORTAL[1])
    r.draw(M["cube"], p @ T(0, 0.05, 0) @ S(3.8, 0.1, 3.8), (0.5, 0.6, 0.65), tex="wall", shin=32, spec=0.5)
    h = 0.6 + 1.6 * k                                                       # pilares crescem ao ativar (animacao por evento)
    for i in range(12):
        a = i * math.pi / 6 + (g.spin * 0.5 if k > 0 else 0)
        r.draw(M["cube"], p @ Ry(a) @ T(1.6, 0.1 + h / 2, 0) @ S(0.25, h, 0.25), col, emis=emis, shin=48, spec=0.8)
    r.draw(M["circle"], p @ T(0, 0.15 + h, 0) @ S(1.6, 1, 1.6), lerp((0.2, 0.4, 0.3), (0.3, 1, 0.5), k), unlit=True)
    if k > 0.001:
        r.draw(M["cube"], p @ T(0, 1.0, 0) @ Ry(g.spin * 2) @ S(2.2 * k, 0.04, 2.2 * k), col, emis=emis, unlit=True)

def draw_scene(r, g, M):
    cube = M["cube"]
    r.draw(M["floor"], np.eye(4), (1, 1, 1), tex="floor", shin=24, spec=0.35)
    r.draw(M["grid"], np.eye(4), (0.05, 0.35, 0.45), unlit=True)
    wall = dict(color=(0.9, 0.95, 1.0), tex="wall", shin=32, spec=0.5)
    w = HALF + 0.5
    for (x, z, sx, sz) in ((0, w, 2 * HALF + 2, 1), (0, -w, 2 * HALF + 2, 1), (w, 0, 1, 2 * HALF), (-w, 0, 1, 2 * HALF)):
        r.draw(cube, T(x, 1.5, z) @ S(sx, 3, sz), **wall)
    for cx, cz, rad, var, col in CORALS: draw_coral(r, M, cx, cz, rad, var, col)   # obstaculos fixos
    for f in g.fish: draw_fish(r, g, M, f)                                          # obstaculos moveis
    draw_portal(r, g, M)
    for c in g.cells:                                                       # celulas de energia
        if c["state"] == 2: continue
        if c["state"] == 0:
            s, y, ang = 1.0, 1.2 + CELL_BOB_AMP * math.sin(CELL_BOB_FREQ * g.t + c["ph"]), g.spin + c["ph"]   # flutuacao periodica
        else:
            u = c["timer"] / COLLECT_TIME                                   # ANIMACAO DE EVENTO: encolhe e gira rapido
            s, y, ang = max(1 - u, 0.02), c["y0"] + 0.6 * u, g.spin * 6
        r.draw(M["octa"], T(c["x"], y, c["z"]) @ Ry(ang) @ S(0.4 * s * CELL_SCALE, 0.6 * s * CELL_SCALE, 0.4 * s * CELL_SCALE),
               (1.0, 0.8, 0.1), emis=(0.55, 0.38, 0.0), shin=64, spec=1.0)
    draw_drone(r, g, M)                                                     # desenhado com z-buffer: oclusao correta

# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def save_screenshot(w, h):
    """Salva o quadro atual em capturas/captura_AAAAMMDD_HHMMSS.png (PNG escrito com zlib, sem bibliotecas extras)."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")
    os.makedirs(folder, exist_ok=True)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    flat = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else np.asarray(data, dtype=np.uint8).reshape(-1)
    img = flat.reshape(h, w, 3)[::-1]                                       # OpenGL le de baixo para cima
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    name = os.path.join(folder, time.strftime("captura_%Y%m%d_%H%M%S.png"))
    with open(name, "wb") as fh:
        fh.write(png)
    print("Captura salva em", name)

def main():
    if not glfw.init(): sys.exit("Falha ao iniciar o GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3); glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE); glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(1100, 700, TITLE, None, None)
    if not win: glfw.terminate(); sys.exit("Falha ao criar a janela (OpenGL 3.3 necessario)")
    glfw.make_context_current(win); glfw.swap_interval(1)

    glEnable(GL_DEPTH_TEST)                                                 # TESTE DE PROFUNDIDADE
    glClearColor(0.01, 0.05, 0.10, 1.0)
    r = Renderer()
    upload_texture(make_floor_tex(), 0); upload_texture(make_wall_tex(), 1)
    M = dict(cube=Mesh(make_cube(), GL_TRIANGLES), octa=Mesh(make_octa(), GL_TRIANGLES),
             sph=Mesh(make_sphere(), GL_TRIANGLES), cone=Mesh(make_cone(), GL_TRIANGLES),
             torus=Mesh(make_torus(), GL_TRIANGLES),
             floor=Mesh(make_floor(HALF, 2 * HALF / 4), GL_TRIANGLE_STRIP), grid=Mesh(make_grid(HALF, 2.0), GL_LINES),
             circle=Mesh(make_circle(), GL_LINE_LOOP))

    game = Game()
    cam = dict(yaw=-math.pi / 4, pitch=0.55, dist=9.0)
    st = dict(drag=False, lx=0.0, ly=0.0, compare=False, shot=False)

    def rot_cam(dyaw):                      # girar a camera tambem gira o drone: e assim que se "mira" a lanterna
        cam["yaw"] += dyaw; game.yaw += dyaw

    def on_key(w, key, sc, action, mods):
        if action != glfw.PRESS: return
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
        elif key == glfw.KEY_L: game.light = not game.light
        elif key == glfw.KEY_R: game.reset(); cam["yaw"] = -math.pi / 4
        elif key == glfw.KEY_C: st["compare"] = not st["compare"]   # comparacao luz ligada x desligada
        elif key == glfw.KEY_P: st["shot"] = True                    # salva captura de tela

    def on_button(w, b, action, mods):
        if b == glfw.MOUSE_BUTTON_LEFT:
            st["drag"] = action == glfw.PRESS; st["lx"], st["ly"] = glfw.get_cursor_pos(w)

    def on_cursor(w, x, y):
        if st["drag"]:
            rot_cam(-(x - st["lx"]) * 0.006)
            cam["pitch"] = min(max(cam["pitch"] + (y - st["ly"]) * 0.006, 0.15), 1.4)
        st["lx"], st["ly"] = x, y

    def on_scroll(w, dx, dy): cam["dist"] = min(max(cam["dist"] - dy, 4.0), 18.0)

    glfw.set_key_callback(win, on_key); glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor); glfw.set_scroll_callback(win, on_scroll)

    last, acc, frames, fps = glfw.get_time(), 0.0, 0, 0
    while not glfw.window_should_close(win):
        now = glfw.get_time(); dt = min(now - last, 0.05); last = now        # DELTA TIME (limitado a 50 ms)
        glfw.poll_events()
        k = lambda key: glfw.get_key(win, key) == glfw.PRESS
        rot_cam((k(glfw.KEY_LEFT) - k(glfw.KEY_RIGHT)) * 1.8 * dt)
        cam["pitch"] = min(max(cam["pitch"] + (k(glfw.KEY_UP) - k(glfw.KEY_DOWN)) * 1.2 * dt, 0.15), 1.4)
        game.update(dt, win, cam["yaw"])

        fw, fh = glfw.get_framebuffer_size(win)
        if fw == 0 or fh == 0: continue
        target = np.array([game.pos[0], 0.8, game.pos[1]])
        cp, cy, d = math.cos(cam["pitch"]), math.sin(cam["pitch"]), cam["dist"]
        eye = target + d * np.array([cp * math.sin(cam["yaw"]), cy, cp * math.cos(cam["yaw"])])
        lamp_pos, lamp_dir = game.lamp()
        view = look_at(eye, target, np.array([0.0, 1.0, 0.0]))
        glEnable(GL_SCISSOR_TEST)
        if st["compare"]:                          # MODO COMPARACAO (tecla C): esquerda = luz LIGADA | direita = luz DESLIGADA
            half = fw // 2
            passes = ((0, half, True), (half, fw - half, False))
        else:
            passes = ((0, fw, game.light),)
        for x0, w_, on in passes:                  # mesma cena e mesma camera, so muda a luz
            glViewport(x0, 0, w_, fh); glScissor(x0, 0, w_, fh)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            game.light_draw = on
            r.frame(view, perspective(math.radians(60), w_ / fh, 0.1, 100.0), eye, on, lamp_pos, lamp_dir)
            draw_scene(r, game, M)
        if st["compare"]:                          # divisoria branca entre as duas metades
            glScissor(fw // 2 - 1, 0, 3, fh); glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT)
            glClearColor(0.01, 0.05, 0.10, 1.0)
        glDisable(GL_SCISSOR_TEST)
        if st["shot"]:
            save_screenshot(fw, fh); st["shot"] = False
        glfw.swap_buffers(win)

        acc += dt; frames += 1                                               # HUD no titulo da janela
        if acc >= 0.25:
            fps = frames / acc; acc, frames = 0.0, 0
            status = ("MISSAO CUMPRIDA! (R reinicia)" if game.won else
                      "Portal ATIVO - va ate ele!" if game.portal_t >= 1 else "Colete as celulas")
            lamp_txt = "ESQ ON | DIR OFF (comparacao)" if st["compare"] else ("ON" if game.light else "OFF")
            glfw.set_window_title(win, f"{TITLE} | Celulas: {game.collected}/{len(CELLS)} | "
                                       f"Luz: {lamp_txt} | {status} | {fps:.0f} FPS")
    glfw.terminate()

if __name__ == "__main__":
    main()
