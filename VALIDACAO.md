# Validação do Ofuscador Elite 1.3.0 — remoção local de legendas

Data: 2 de agosto de 2026

## Windows 10/11 x64

- Pacote: `OfuscadorElite-Windows-x64-v1.3.0.zip`
- Tamanho: 268.829.083 bytes
- SHA-256: `7101A75014AFFB8AE29E1AE8B9495F9ECE4379F63A58DFDDCE5E0F05C35A4558`
- O ZIP passou na leitura completa de CRC, foi extraído em pasta nova e o executável extraído concluiu o autoteste com código 0, sem Python ou Node.js externos.
- O autoteste confirmou interface, API, FFmpeg, FFprobe, libx264, Piper, ONNX Runtime, OpenCV e RapidOCR.

## Pacote de IA

- Pacote: `OfuscadorElite-IA-Legendas-v1.zip`
- Tamanho: 217.238.607 bytes
- SHA-256: `E655D34870D18A1F6535A61D14C93D807A982CCA8DE78231B2135149D9C6774D`
- A instalação em diretório temporário protegido foi concluída.
- RapidOCR detectou texto em um quadro de diagnóstico.
- LaMa ONNX reconstruiu a área mascarada e retornou uma imagem válida.
- Um vídeo sintético percorreu a pipeline completa: H.264, resolução preservada, áudio preservado, nenhuma faixa de legenda e duração dentro da tolerância de um quadro.

## Testes realizados

- 30 testes do backend passaram.
- 7 testes da interface passaram.
- A verificação TypeScript e a compilação portátil passaram.
- Foram testados: faixa separada sem recodificação, análise de metadados, modo combinado com antifase, regiões e intervalos, recuperação temporal e rejeição de caminhos maliciosos no ZIP.
- A pipeline existente de voz/antifase foi reexecutada e manteve a correlação negativa do original e a voz em fase.
- O construtor Windows agora espera corretamente o executável gráfico terminar, valida o ZIP criado e registra SHA-256 antes de declarar sucesso.

## Limitações conhecidas

- A remoção de legenda gravada reconstrói uma estimativa do fundo; não recupera pixels ocultos com garantia exata.
- HDR fica bloqueado nesta primeira versão.
- VFR é convertido para FPS constante equivalente quando a imagem precisa ser reconstruída.
- A URL HTTPS definitiva da Hostinger ainda não foi informada. Até ela ser configurada no construtor, o app baixa o LaMa da fonte pública indicada no código ou aceita o ZIP manualmente.

## macOS

O construtor `build-macos.command` foi atualizado para gerar arm64 e Intel, verificar Rosetta 2, libx264, RapidOCR, ONNX Runtime e LaMa. A validação integral dos aplicativos macOS ainda precisa ser executada em um Mac Apple Silicon com Rosetta 2, conforme previsto no plano.
