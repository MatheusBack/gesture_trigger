# Visual Pattern Gesture Trigger

Abre a webcam, encontra a sua mão e, quando você faz um gesto (punho, joinha, sinal de paz...),
**mostra uma imagem na tela** e avisa no terminal qual gesto foi reconhecido.

## Como funciona

```text
webcam → HandDetector → FingerStateEstimator → PatternMatcher → overlay na tela
         (MediaPipe)     (geometria 3D)         (patterns.json)   (imagem do gesto)
```

1. **HandDetector** (`core/hand_detector.py`): o MediaPipe HandLandmarker acha 21 pontos 3D na mão.
2. **FingerStateEstimator** (`core/finger_state.py`): decide, para cada dedo, se está esticado (1) ou
   dobrado (0), usando *ângulos das articulações* e *distâncias normalizadas pela palma*. Isso
   funciona com a mão inclinada, girada, mais perto/longe, esquerda ou direita. Médias e histerese
   evitam que um dedo "pisque" entre 0 e 1.
3. **PatternMatcher** (`core/pattern_matcher.py`): compara `[polegar, indicador, médio, anelar, mínimo]`
   com `patterns.json`. O gesto só é confirmado após alguns quadros seguidos (e só some após alguns
   quadros sem ele), então não há piscadas.
4. **GestureApp** (`core/app.py`) desenha o esqueleto (verde = esticado, laranja = dobrado), o nome
   do gesto e a imagem com fade-in.

## Instalação (Windows)

Precisa de Python 3.10+ de verdade (não o atalho da Microsoft Store; instale de python.org marcando
"Add Python to PATH", ou `winget install Python.Python.3.12`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/generate_assets.py      # só se a pasta assets/ estiver sem as imagens
python main.py
```

Se o PowerShell bloquear o venv: `Set-ExecutionPolicy -Scope Process RemoteSigned`.
Na **primeira execução** o modelo da mão (~8 MB) é baixado para `models/`. Sem internet, baixe a URL
que o programa mostrar e salve no caminho indicado.

## Uso

```powershell
python main.py                 # webcam padrão
python main.py --camera 1      # outra webcam
python main.py --debug         # já abre com o painel de notas dos dedos
```

Teclas: **Q/ESC** sair · **D** painel de debug · **L** liga/desliga o esqueleto.

### Se um gesto não for reconhecido

Aperte **D**. As barras mostram a nota (0 a 1) de cada dedo e os dois limites (linhas brancas):
abaixo do primeiro = dobrado, acima do segundo = esticado. Veja qual dedo fica "no meio" e
ajuste os números no topo de `core/finger_state.py`. Dicas: boa iluminação frontal, fundo sem
muita bagunça, mão a ~40-80 cm da câmera e inteira dentro do quadro.

## Configurando gestos

Edite `patterns.json` (não precisa mexer no código):

```json
{ "name": "Sinal de Paz", "pattern": [null, 1, 1, 0, 0], "image": "assets/paz.png" }
```

- `pattern` = `[polegar, indicador, médio, anelar, mínimo]`: `1` esticado, `0` dobrado, `null` tanto faz.
- Se vários gestos combinam por causa de `null`, vence o mais específico (menos `null`s).
- Dois gestos com o mesmo padrão são rejeitados na carga.
- `image`: caminho do PNG. `python tools/generate_assets.py` cria um cartão ilustrado para cada
  gesto que ainda não tem imagem; para usar uma sua, salve o PNG no caminho (veja `assets/README.md`).

## Testes

```powershell
python -m unittest discover -s tests -v
```

90 testes, sem webcam. Cobrem a geometria dos dedos (inclusive mão girada e com ruído, usando uma
mão sintética em `tests/synthetic_hand.py`), o comparador de padrões, o overlay, a integridade de
`patterns.json` + imagens, o fluxo completo gesto → imagem na tela, e o loop do `main.py`.

## Limitações conhecidas

- O reconhecimento usa só *quais dedos estão esticados*: gestos que diferem por orientação ou
  contato entre dedos (muitas letras do alfabeto de Libras) não são distinguíveis ainda. O joinha
  e o "polegar para baixo" também são iguais.
- O polegar é o dedo mais difícil; por isso a maioria dos gestos padrão usa `null` nele.
- Uma mão por vez.
