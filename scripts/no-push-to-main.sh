#!/usr/bin/env sh
# Bloqueia 'git push' que atualize diretamente a branch main.
# Rodado pelo pre-commit no stage pre-push (ver .pre-commit-config.yaml).
# Contorno de emergencia, sob sua responsabilidade: git push --no-verify
set -eu

PROTEGIDA="refs/heads/main"

alvo="${PRE_COMMIT_REMOTE_BRANCH:-}"
if [ -z "$alvo" ]; then
    # pre-commit antigo nao exporta a ref remota: cai para a branch atual
    atual="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
    [ "$atual" = "main" ] && alvo="$PROTEGIDA"
fi

if [ "$alvo" = "$PROTEGIDA" ]; then
    echo "-------------------------------------------------------------------"
    echo " Push direto para 'main' bloqueado."
    echo " Crie um branch e abra um Pull Request:"
    echo "     git switch -c minha-tarefa"
    echo "     git push -u origin minha-tarefa"
    echo " (emergencia, sob sua responsabilidade: git push --no-verify)"
    echo "-------------------------------------------------------------------"
    exit 1
fi

exit 0
