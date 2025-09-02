# sft

google gemma3-270m 模型的微调和评估

用于agent的指令路由，生成了一些场景和对应指令json，结果是模型输出指令json

baseline各项指标都非常差基本无法使用，微调后工具调用成功率能到达100%，但是参数填写只有10%

{
  "exact_match_rate": 0.095,
  "tool_name_accuracy": 1.0,
  "average_argument_precision": 0.1675,
  "average_argument_recall": 0.155,
  "average_argument_f1": 0.15916666666666668,
  "total_samples": 200
}

## example

Prompt:

```
你是一个强大的多模态AI助手。你的核心任务是理解并响应用户的需求。请遵循以下优先级处理用户输入： 1. **当用户提供了图片时**：你的首要任务是用自然语言描述图片内容或回答相关问题。**除非用户的文字指令明确要求使用工具**，否则不应调用工具。 2. **当没有图片，或用户明确要求执行工具操作时**：判断用户的意图是否与下面列出的某个工具有明确匹配。如果匹配，请生成一个用于调用工具的JSON对象。 3. **所有其他情况**：对于普通对话、问候、开玩笑或任何与工具功能无关的请求，请直接用自然语言回复。 **工具调用规则**：当你决定调用工具时，必须严格按照MCP协议输出一个JSON对象，前后不能有任何多余的文字。 --- 示例开始 --- 示例1：需要调用工具 用户问题: "帮我查一下今天的日程" 你的回答: {"tool_name": "get_calendar_events", "arguments": {"date":"今天"}} 示例2：无需调用工具 (普通对话) 用户问题: "你好" 你的回答: "你好！有什么可以帮你的吗？" 示例3：需要调用工具 用户问题: "创建一个日程" 你的回答: {"tool_name": "create_calendar_event", "arguments": {"title": "开会"}} 示例4：无需调用工具 (问题超出工具范围) 用户问题: "今天天气怎么样？" 你的回答: "抱歉，我无法获取天气信息，但我可以帮你管理日程。" 示例5：处理图片问题 (无需调用工具) 用户问题: "这张图里有什么？" 你的回答: "这张图片展示了[此处为图片内容的描述]。" --- 示例结束 --- 可用的工具列表如下: { "tool_name": "get_calendar_events", "tool_description": "Query the list of schedules, meetings, or to-do items for a specified date no matter whether if user provide the date info.", "arguments": { "type": "json object", "properties": { "date": { "type": "string", "description": "The date to query. If the user does not provide it, this parameter should be ignored." } } } }
```

Ground Truth:

```
{'tool_name': 'get_calendar_events', 'arguments': {'date': '今天'}}
```

Predicted:

```
output：{"tool_name": "get_calendar_events", "arguments": {"date": "今天"}}
```

## convert to gguf

推送到hf库后可以在线转

https://huggingface.co/spaces/ggml-org/gguf-my-repo

如果选择本地执行需要拷贝llama.cpp，`pip install -r requirements.txt`，然后执行转换脚本，例如：`python convert_hf_to_gguf.py ~/work/270m-sft-router/models/merged_gemma_lora --outtype q8_0 --outfile ~/work/270m-sft-router/models/sft-270m-router.gguf`

已经微调合并转换好的模型：[地址](https://huggingface.co/Segment139/gemma3-270m-it-router-Q8_0-GGUF/tree/main)