import json
import random
import csv

# Base prompt template, copied from the user's request
BASE_PROMPT = """你是一个强大的多模态AI助手。你的核心任务是理解并响应用户的需求。请遵循以下优先级处理用户输入：
1.  **当用户提供了图片时**：你的首要任务是用自然语言描述图片内容或回答相关问题。**除非用户的文字指令明确要求使用工具**，否则不应调用工具。
2.  **当没有图片，或用户明确要求执行工具操作时**：判断用户的意图是否与下面列出的某个工具有明确匹配。如果匹配，请生成一个用于调用工具的JSON对象。
3.  **所有其他情况**：对于普通对话、问候、开玩笑或任何与工具功能无关的请求，请直接用自然语言回复。
**工具调用规则**：当你决定调用工具时，必须严格按照MCP协议输出一个JSON对象，前后不能有任何多余的文字。
--- 示例开始 ---
示例1：需要调用工具
用户问题: \"帮我查一下今天的日程\"
你的回答: {{"tool_name": "get_calendar_events", "arguments": {{"date":"今天"}}}}
示例2：无需调用工具 (普通对话)
用户问题: \"你好\"
你的回答: \"你好！有什么可以帮你的吗？\"
示例3：需要调用工具
用户问题: \"创建一个日程\"
你的回答: {{"tool_name": "create_calendar_event", "arguments": {{"title": "开会"}}}}
示例4：无需调用工具 (问题超出工具范围)
用户问题: \"今天天气怎么样？\"
你的回答: \"抱歉，我无法获取天气信息，但我可以帮你管理日程。\"
示例5：处理图片问题 (无需调用工具)
用户问题: \"这张图里有什么？\"
你的回答: \"这张图片展示了[此处为图片内容的描述]。\"
--- 示例结束 ---
可用的工具列表如下:
{tool_definition}"""

# Expanded tool library for data generation
TOOLS = {
    "create_calendar_event": {
        "definition": {
            "tool_name": "create_calendar_event",
            "tool_description": "Create a new calendar event, meeting, or to-do item.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "title": {"type": "string", "description": "title or theme of the event/meeting"},
                    "start_time": {"type": "string", "description": "start time of the event. if the user does not provide then ignore this parameter because the create calendar menu will let user choose the time info"}
                },
                "required": ["title"]
            }
        },
        "arguments_samples": {
            "title": ["项目评审会", "团队午餐", "和张医生预约", "支付信用卡账单", "写周报"],
            "start_time": ["明天上午10点", "今天下午3:30", "周五晚上7点", "2025-10-10 09:00", None]
        }
    },
    "get_calendar_events": {
        "definition": {
            "tool_name": "get_calendar_events",
            "tool_description": "Query the list of schedules, meetings, or to-do items for a specified date no matter whether if user provide the date info.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "date": {"type": "string", "description": "The date to query. If the user does not provide it, this parameter should be ignored."}
                }
            }
        },
        "arguments_samples": {
            "date": ["今天", "明天", "后天", "本周", "下周三", "2025-09-05"]
        }
    },
    "send_message": {
        "definition": {
            "tool_name": "send_message",
            "tool_description": "Send a message to a contact.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "recipient": {"type": "string", "description": "The name or phone number of the person to send the message to."}, 
                    "content": {"type": "string", "description": "The content of the message."}
                },
                "required": ["recipient", "content"]
            }
        },
        "arguments_samples": {
            "recipient": ["王伟", "李娜", "项目经理", "13912345678"],
            "content": ["下午的会议改到301会议室了", "我晚上会晚点到家", "记得查收邮件", "生日快乐！"]
        }
    },
    "set_reminder": {
        "definition": {
            "tool_name": "set_reminder",
            "tool_description": "Set a reminder for a specific task or event.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "task": {"type": "string", "description": "The task to be reminded of."},
                    "time": {"type": "string", "description": "The time for the reminder."}
                },
                "required": ["task", "time"]
            }
        },
        "arguments_samples": {
            "task": ["取快递", "给妈妈打电话", "买菜", "提交报销单"],
            "time": ["下午5点", "15分钟后", "明天早上9点半", "2025-09-02 18:00"]
        }
    },
    "get_weather": {
        "definition": {
            "tool_name": "get_weather",
            "tool_description": "Get the weather forecast for a specific location.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "location": {"type": "string", "description": "The city name to get the weather for."},
                    "date": {"type": "string", "description": "The date for the forecast, e.g., '今天' or '明天'."}
                },
                "required": ["location"]
            }
        },
        "arguments_samples": {
            "location": ["北京", "上海", "广州", "深圳", "杭州"],
            "date": ["今天", "明天", "后天", None]
        }
    },
    "search_web": {
        "definition": {
            "tool_name": "search_web",
            "tool_description": "Search the web for information.",
            "arguments": {
                "type": "json object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        },
        "arguments_samples": {
            "query": ["最新的AI研究进展", "如何学习弹吉他", "附近有什么好吃的餐厅", "一部高分悬疑电影推荐"]
        }
    }
}

def generate_data(num_samples):
    data = []
    tool_names = list(TOOLS.keys())
    for _ in range(num_samples):
        tool_name = random.choice(tool_names)
        tool_info = TOOLS[tool_name]
        
        tool_definition_str = json.dumps(tool_info["definition"], ensure_ascii=False, indent=2)
        
        input_prompt = BASE_PROMPT.format(tool_definition=tool_definition_str)
        
        args = {}
        for arg_name, samples in tool_info["arguments_samples"].items():
            is_required = arg_name in tool_info["definition"]["arguments"].get("required", [])
            # Ensure required arguments are always present
            # For optional args, include them with a 70% probability
            if is_required or random.random() < 0.7:
                value = random.choice(samples)
                if value is not None:
                    args[arg_name] = value
        
        # Ensure at least one argument is present if any are defined
        if not args and tool_info["arguments_samples"]:
            arg_name = random.choice(list(tool_info["arguments_samples"].keys()))
            value = random.choice(tool_info["arguments_samples"].get(arg_name, [None]))
            if value is not None:
                args[arg_name] = value

        tool_call = {
            "tool_name": tool_name,
            "arguments": args
        }
        
        output_str = "output：" + json.dumps(tool_call, ensure_ascii=False)
        
        data.append({
            "text": input_prompt,
            "label": output_str
        })
        
    return data

# Generate 1000 samples
generated_data = generate_data(1000)

# Write to a CSV file
file_path = "./data/finetuning_data.csv"
with open(file_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(generated_data)

print(f"成功生成了1000条微调数据，并已保存到文件：{file_path}")