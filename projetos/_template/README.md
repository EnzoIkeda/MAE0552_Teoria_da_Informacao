# _template

Esqueleto para começar uma proposta nova. **Não é uma entrega.**

```bash
cp -r projetos/_template projetos/0N_nome_curto
```

Depois, dentro da cópia:

- duplique `topico_exemplo/` uma vez por tópico do enunciado e renomeie
  (`topico1_...`, `topico2_...`);
- renomeie o `.ipynb` de cada tópico para casar com a pasta;
- preencha `src/model.py` e `relatorio.md`;
- registre a proposta em `projetos/README.md`.

Cada tópico já vem com: notebook (um passo por seção), `src/` importável do
notebook, `dados/` e `figuras/` ignorados pelo git, e `relatorio.md`.
