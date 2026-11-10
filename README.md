# Abissal-1: O Resgate Submarino

Minijogo 3D em Python + PyOpenGL (OpenGL 3.3 core) com janela GLFW.
Trabalho Prático 1 — Computação Gráfica — IFSULDEMINAS, Campus Machado — Prof. Renan.

## Integrantes

|Nome|
|-|
|*Amanda Dias*|
|*Laura Evelyn*|

## Conteúdo da pasta

```
abissal1.py        código-fonte (arquivo único)
requirements.txt   dependências
README.md          este arquivo
capturas/          criada pelo jogo ao apertar P (capturas de tela para o relatório)
```

Não há imagens externas: as duas texturas (piso e painel) são **geradas por código** (NumPy) em `make\_floor\_tex` e `make\_wall\_tex`.

## Instalação e execução

Requer Python 3.9+ e placa/driver com OpenGL 3.3.

```bash
pip install -r requirements.txt
python abissal1.py
```

## Objetivo

Pilote o drone submarino no escuro, usando a lanterna para achar as **5 células de energia** (amarelas). Desvie dos **3 corais** e dos **3 peixes** e entre no **portal**, que acende em verde neon quando todas as células forem coletadas. O contador e a mensagem de missão aparecem no título da janela.

## Controles

|Tecla / Mouse|Ação|
|-|-|
|W, A, S, D|Mover o drone (relativo à câmera)|
|Setas|Girar/inclinar a câmera (o drone gira junto: é assim que se mira a lanterna)|
|Mouse (botão esquerdo + arrastar)|Mesmo efeito das setas|
|Roda do mouse|Zoom da câmera|
|**L**|**Liga/desliga a luz** (lanterna + luz de apoio do casco)|
|**C**|**Modo comparação**: tela dividida, esquerda com luz LIGADA e direita com luz DESLIGADA|
|P|Salva captura de tela em `capturas/`|
|R|Reinicia a missão|
|Esc|Encerra a aplicação|

## Para a apresentação: alterações rápidas

Todos os parâmetros abaixo ficam no topo de `abissal1.py` (seção "PARAMETROS AJUSTAVEIS").

|Pedido|O que alterar|
|-|-|
|Velocidade de uma animação|`CELL\_SPIN\_DEG` (giro das células), `CELL\_BOB\_FREQ` (flutuação), `SPEED` (drone)|
|Posição da luz|`LAMP\_OFFSET` (x, y, z no referencial do drone; ex.: `(0, 3, 0)` = luz vinda de cima)|
|Intensidade / abertura da luz|`LIGHT\_INTENSITY`, `CONE\_IN\_DEG`, `CONE\_OUT\_DEG`|
|Escala de um objeto|`CELL\_SCALE`, `DRONE\_SCALE`|

## Requisitos × onde estão no código (`abissal1.py`)

|Requisito|Implementação|
|-|-|
|Primitivas|`GL\_TRIANGLES` (casco, corais, peixes, células), `GL\_TRIANGLE\_STRIP` (piso), `GL\_LINES` (grade), `GL\_LINE\_LOOP` (anel do portal)|
|Pipeline local → mundo → câmera → projeção|Docstring do arquivo, `look\_at`, `perspective` e vertex shader|
|Drone composto e hierarquia|`draw\_drone`: casco → braços → anéis dos propulsores → hélices; hélice de popa; leme; sensor frontal (domo) e faróis|
|Translação / Rotação / Escala|`T`, `Ry`, `Rz`, `S`|
|Delta time|`dt` calculado no laço principal e usado em `Game.update`|
|Animações|Rotação contínua (hélices, células), flutuação senoidal das células, cauda dos peixes, coleta (escala → 0) e ativação do portal|
|Obstáculos|3 corais fixos (`draw\_coral`) e 3 peixes móveis (`draw\_fish`), colisão círculo × círculo|
|Texturas|Duas texturas procedurais; UV com `GL\_REPEAT` no piso e projeção por caixa nas paredes (sem deformação)|
|Iluminação|Ambiente + lanterna (spot light) com difusa e especular (Blinn-Phong), normais corrigidas por `uNormalMat`; tecla L; modo comparação (C)|
|Profundidade|`glEnable(GL\_DEPTH\_TEST)`|
|Limites, coleta, contagem, conclusão, reinício|`Game.collide`, `Game.update`, título da janela, tecla R|



