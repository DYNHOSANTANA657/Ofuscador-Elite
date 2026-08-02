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

## Correção posterior — remoção de legenda gravada

O diagnóstico de vídeo citado acima usava um clipe sintético de 8 quadros a 8 fps.
Nessa escala a pipeline passava mesmo quebrada: com vídeo real o processamento parava
em 97% com `A duração do resultado ficou fora da tolerância de um quadro`, e o
aplicativo ainda apagava o resultado e descartava o envio.

Causa: a leitura dos quadros era feita pelo OpenCV e a regravação pelo FFmpeg. São
builds diferentes de libavcodec e o OpenCV pode devolver menos quadros em vídeos com
B-frames, GOP aberto ou taxa variável. Cada quadro perdido encurtava o resultado.

O que mudou:

- A leitura passou a usar o mesmo FFmpeg da gravação, por pipe `rawvideo`, com
  `-fps_mode cfr` normalizando VFR na entrada. A contagem de quadros fecha por
  construção — medido: 600 de 600 em um clipe de 20 s a 30000/1001 com B-frames.
- A verificação final compara duração de vídeo com duração de vídeo. A duração do
  contêiner MP4 costuma ser a da faixa de áudio: no mesmo clipe, 20,400 s de contêiner
  contra 20,020 s de vídeo — cinco vezes a tolerância antiga.
- Fora da tolerância virou aviso, não falha. O vídeo é salvo e o desvio aparece na
  interface com os números.
- Falha não descarta mais o envio nem a análise OCR, e o material já reconstruído é
  mantido como `-com-aviso.mp4`.
- Cada trabalho grava `Dados/logs/job-<id>.txt` com comandos, contagens e durações.
- O diagnóstico `scripts/verify-subtitle-video-pipeline.py` passou a usar 20 s a
  30000/1001 com B-frames, GOP de 250 e áudio mais longo que o vídeo, e confere
  contagem de quadros além da duração.

### Causa raiz confirmada com um vídeo real

Um vídeo de 2 minutos, 720x1280, 30 fps, 3591 quadros passou pela remoção gravada
completa e o log registrou:

```
origem:    contêiner 119.837s · vídeo 119.700s · 3591 quadros
resultado: contêiner 119.722s · vídeo 119.700s · 3591 quadros
```

A verificação antiga comparava contêiner com contêiner: `|119,722 − 119,837| = 0,115s`
contra uma tolerância de `1/30 + 0,04 = 0,073s`. Falhava por 42 ms com o vídeo
intacto — o que mudou foi a duração declarada da faixa de **áudio**, recalculada pelo
FFmpeg ao remuxar. A verificação nova compara vídeo com vídeo: diferença zero.

Os 3591 de 3591 quadros lidos também descartam a hipótese de perda de quadros na
leitura. A troca do OpenCV pelo FFmpeg continua correta como reforço, mas não era
a causa.

## Correção posterior — mancha na reconstrução

A remoção passou a concluir, mas deixava uma mancha visível no lugar da legenda,
contrariando a promessa do LEIA-ME de nunca substituir uma falha por borrão.

Causa: o LaMa rodava apenas no primeiro quadro de cada região. Nos seguintes, o
código deformava com fluxo óptico o quadro **já limpo** anterior. Numa legenda de
vinte segundos isso propagava o mesmo remendo por cerca de 650 quadros, cada um
deformando o erro do anterior até virar um borrão.

Correção: a propagação temporal passou a ter teto de aproximadamente um segundo de
quadros; esgotado o teto, o LaMa roda de novo. Medido num trecho de 20 s do mesmo
vídeo, com 600 quadros:

- antigo: 600 quadros em 243 s (2,47 quadros/s), com mancha
- novo: 600 quadros em 371 s (1,62 quadros/s), sem mancha

O custo é cerca de 1,5x mais tempo de processamento.

## macOS

O construtor `build-macos.command` foi atualizado para gerar arm64 e Intel, verificar Rosetta 2, libx264, RapidOCR, ONNX Runtime e LaMa. A validação integral dos aplicativos macOS ainda precisa ser executada em um Mac Apple Silicon com Rosetta 2, conforme previsto no plano.
