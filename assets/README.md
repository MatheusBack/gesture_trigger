# Assets

Imagens exibidas na tela para cada gesto de `patterns.json` (campo `image`).
PNG com fundo transparente é suportado.

- Gerar as imagens padrão (só as que faltam): `python tools/generate_assets.py`
- Recriar todas, sobrescrevendo: `python tools/generate_assets.py --force`
- Usar imagem sua (foto de um sinal em Libras, ícone...): salve um PNG no caminho
  indicado em `image` e **não** use `--force`. Se a imagem faltar, o programa não
  trava: avisa no terminal e na tela.
