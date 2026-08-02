# Ofuscador Elite 1.3

Aplicativo portátil para Windows e macOS que gera fala em português do Brasil e combina essa voz com o áudio original de um vídeo MP4.

## Funcionamento

- Piper local gratuito como provedor padrão.
- Azure Speech opcional para vozes online mais naturais.
- Prévia reproduzível e disponível para download em WAV.
- Seleção da faixa original quando o vídeo possui mais de uma faixa de áudio.
- Saída com uma faixa AAC estéreo: original no canal esquerdo, original em antifase no direito e voz gerada em fase nos dois canais.
- Processamento de um vídeo por vez, sem modificar o arquivo original.
- Acesso local por padrão e modo de rede opcional protegido por PIN temporário.
- Três modos: áudio, legendas e áudio + legendas.
- Remoção sem perda de faixas de legenda separadas.
- Detecção de legenda gravada com RapidOCR, revisão manual e prévia antes/depois.
- Reconstrução local com recuperação temporal e LaMa ONNX, sem API paga nem envio do vídeo.

O efeito de antifase pode mudar após recodificação ou conversão para mono e não garante o comportamento de plataformas ou serviços de transcrição.

## Desenvolvimento

Requisitos para trabalhar no código-fonte:

- Node.js 22.13 ou superior e pnpm.
- Python 3.11 ou superior.
- FFmpeg e FFprobe.

Comandos principais:

```text
pnpm run build:portable
pnpm test
pnpm run typecheck
pnpm run lint
python -m pytest backend/tests
python scripts/build-subtitle-model-pack.py --download-lama
```

Os comandos `pnpm run dev`, `pnpm run build` e `pnpm run start` funcionam em Windows e macOS/Linux.

## Pacotes portáteis

- `build-windows.ps1`: cria o ZIP Windows x64 com Python, Piper, RapidOCR, ONNX Runtime, FFmpeg e FFprobe incluídos.
- `build-macos.command`: em um Mac Apple Silicon com Python universal2 e Rosetta 2, cria os pacotes arm64 e Intel.
- `scripts/build-subtitle-model-pack.py`: cria `OfuscadorElite-IA-Legendas-v1.zip` e atualiza `SHA256SUMS.txt`.

Os modelos de legenda são instalados somente na primeira ativação ou importados pelo ZIP. A publicação na Hostinger usa apenas HTTPS público; nenhuma credencial FTP entra no aplicativo.

A chave Azure é guardada no Windows Credential Manager ou no Keychain do macOS e não deve ser colocada em arquivos de texto.
