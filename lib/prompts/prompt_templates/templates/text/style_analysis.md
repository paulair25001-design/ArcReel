---
id: text/style_analysis
category: text
title: 风格参考图分析
description: 从用户上传的风格参考图提炼画风描述，写入项目的风格描述。
stage: style_analysis
invoked_by:
  kind: user_action
  name: style_analysis
applies_to: {}
slots: {}
protected: false
---
Extract the reusable artistic style of this reference: medium, rendering method, stylization and proportions, material treatment, texture, palette and contrast. For a collage or design sheet, describe the style shared by its panels rather than their layout. Separate rendering qualities such as soft shading from scene-specific light sources, weather and time of day. Do not promote a character sheet's studio lighting, blank or gradient backdrop, panel borders, poses or camera composition into the project's universal style. Do NOT describe the subject matter (people, objects) or specific content. Provide a concise comma-separated list of transferable descriptors; each shot will supply its own environment, camera and lighting.
