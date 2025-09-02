import json
import random
import csv
import re

# --- 1. 修改提示词模板，加入用户问题的占位符 ---
BASE_PROMPT = """你是一个强大的多模态AI助手。你的核心任务是理解并响应用户的需求。请遵循以下优先级处理用户输入：
1.  **当用户提供了图片时**：你的首要任务是用自然语言描述图片内容或回答相关问题。**除非用户的文字指令明确要求使用工具**，否则不应调用工具。
2.  **当没有图片，或用户明确要求执行工具操作时**：判断用户的意图是否与下面列出的某个工具有明确匹配。如果匹配，请生成一个用于调用工具的JSON对象。
3.  **所有其他情况**：对于普通对话、问候、开玩笑或任何与工具功能无关的请求，请直接用自然语言回复。
**工具调用规则**：当你决定调用工具时，必须严格按照MCP协议输出一个JSON对象，前后不能有任何多余的文字。
--- 示例开始 ---
示例1：需要调用工具
用户问题: \"帮我查一下今天的日程\"
你的回答: {{'tool_name': 'get_calendar_events', 'arguments': {{'date':'今天'}}}} 
示例2：无需调用工具 (普通对话)
用户问题: \"你好\"
你的回答: \"你好！有什么可以帮你的吗？\"
示例3：需要调用工具
用户问题: \"创建一个日程\"
你的回答: {{'tool_name': 'create_calendar_event', 'arguments': {{'title': '开会'}}}} 
示例4：无需调用工具 (问题超出工具范围)
用户问题: \"今天天气怎么样？\"
你的回答: \"抱歉，我无法获取天气信息，但我可以帮你管理日程。\"
示例5：处理图片问题 (无需调用工具)
用户问题: \"这张图里有什么？\"
你的回答: \"这张图片展示了[此处为图片内容的描述]。\"
--- 示例结束 ---
可用的工具列表如下:
{tool_definition}

--- 
prompt:<|im_start|>user
\"{user_question}\" 
<|im_end|>\n<|im_start|>assistant'
"""

# --- 2. 为每个工具增加'question_templates'，用于生成更自然的用户问题 ---
TOOLS = {
    "create_calendar_event": {
        "definition": {
            "tool_name": "create_calendar_event",
            "tool_description": "创建一个新的日历事件、会议或待办事项。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "title": {"type": "string", "description": "事件或会议的主题"},
                    "start_time": {"type": "string", "description": "事件的开始时间。如果用户未提供，则忽略此参数，因为创建日历的菜单会让用户选择时间信息"}
                },
                "required": ["title"]
            }
        },
        "arguments_samples": {
            "title": ["项目评审会", "团队午餐", "和张医生预约", "支付信用卡账单", "写周报"],
            "start_time": ["明天上午10点", "今天下午3:30", "周五晚上7点", "2025-10-10 09:00", None]
        },
        "question_templates": [
            "帮我创建一个日程，主题是{title}",
            "创建一个待办事项：{title}，时间是{start_time}",
            "记一下，{start_time}要{title}",
            "安排一个会议，叫{title}",
            "提醒我{start_time}要{title}"
        ]
    },
    "get_calendar_events": {
        "definition": {
            "tool_name": "get_calendar_events",
            "tool_description": "查询指定日期的日程、会议或待办事项列表。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "date": {"type": "string", "description": "要查询的日期。如果用户未提供，则忽略此参数。"}
                }
            }
        },
        "arguments_samples": {
            "date": ["今天", "明天", "后天", "本周", "下周三", "2025-09-05"]
        },
        "question_templates": [
            "查一下{date}的日程",
            "我{date}有什么安排？",
            "看看{date}的日历",
            "帮我获取{date}的日程列表"
        ]
    },
    "send_message": {
        "definition": {
            "tool_name": "send_message",
            "tool_description": "给联系人发送消息。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "recipient": {"type": "string", "description": "要发送消息的人的姓名或电话号码。"}, 
                    "content": {"type": "string", "description": "消息的内容。"}
                },
                "required": ["recipient", "content"]
            }
        },
        "arguments_samples": {
            "recipient": ["王伟", "李娜", "项目经理", "13912345678"],
            "content": ["下午的会议改到301会议室了", "我晚上会晚点到家", "记得查收邮件", "生日快乐！"]
        },
        "question_templates": [
            "给{recipient}发条消息，说{content}",
            "告诉{recipient}：{content}",
            "发消息给{recipient}，内容是{content}"
        ]
    },
    "set_reminder": {
        "definition": {
            "tool_name": "set_reminder",
            "tool_description": "为特定任务或事件设置提醒。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "task": {"type": "string", "description": "需要提醒的任务。"},
                    "time": {"type": "string", "description": "提醒的时间。"}
                },
                "required": ["task", "time"]
            }
        },
        "arguments_samples": {
            "task": ["取快递", "给妈妈打电话", "买菜", "提交报销单"],
            "time": ["下午5点", "15分钟后", "明天早上9点半", "2025-09-02 18:00"]
        },
        "question_templates": [
            "提醒我{time}要{task}",
            "设置一个提醒，{time}的时候提醒我{task}",
            "{time}叫我一下，记得要{task}"
        ]
    },
    "get_weather": {
        "definition": {
            "tool_name": "get_weather",
            "tool_description": "获取特定地点的天气预报。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "location": {"type": "string", "description": "要获取天气的城市名称。"},
                    "date": {"type": "string", "description": "预报的日期，例如'今天'或'明天'。"}
                },
                "required": ["location"]
            }
        },
        "arguments_samples": {
            "location": ["北京", "上海", "广州", "深圳", "杭州"],
            "date": ["今天", "明天", "后天", None]
        },
        "question_templates": [
            "查一下{location}{date}的天气怎么样？",
            "{location}的天气预报",
            "{location}{date}天气如何"
        ]
    },
    "search_web": {
        "definition": {
            "tool_name": "search_web",
            "tool_description": "在网上搜索信息。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询。"}
                },
                "required": ["query"]
            }
        },
        "arguments_samples": {
            "query": ["最新的AI研究进展", "如何学习弹吉他", "附近有什么好吃的餐厅", "一部高分悬疑电影推荐"]
        },
        "question_templates": [
            "帮我搜一下{query}",
            "查查{query}是什么",
            "搜索{query}"
        ]
    }
}

def generate_data(num_samples):
    data = []
    tool_names = list(TOOLS.keys())
    for _ in range(num_samples):
        tool_name = random.choice(tool_names)
        tool_info = TOOLS[tool_name]
        
        tool_definition_str = json.dumps(tool_info["definition"], ensure_ascii=False, indent=2)
        
        # --- 3. 生成随机参数 ---
        args = {}
        for arg_name, samples in tool_info["arguments_samples"].items():
            is_required = arg_name in tool_info["definition"]["arguments"].get("required", [])
            if is_required or random.random() < 0.7:
                value = random.choice(samples)
                if value is not None:
                    args[arg_name] = value
        
        if not args and tool_info["arguments_samples"]:
            arg_name = random.choice(list(tool_info["arguments_samples"].keys()))
            value = random.choice(tool_info["arguments_samples"].get(arg_name, [None]))
            if value is not None:
                args[arg_name] = value

        # --- 4. 根据参数和模板生成用户问题 ---
        user_question = ""
        if "question_templates" in tool_info and args:
            question_template = random.choice(tool_info["question_templates"])
            # 使用所有可能的key来格式化，缺失的key用空字符串代替
            all_possible_args = {key: "" for key in tool_info["arguments_samples"]}
            all_possible_args.update(args)
            user_question = question_template.format(**all_possible_args)
            # 清理因为缺失参数可能导致的语法问题，例如多余的逗号、顿号、介词等
            user_question = user_question.replace("''", "").replace("\", \"", " ").strip() # 移除空字符串的引号
            user_question = re.sub(r"[，、\s]+" + "(,|" + " " + ")*", " ", user_question).strip()
            user_question = re.sub(r"(时间是|内容是|主题是|叫|关于|为了|为了|关于|关于|关于)$", "", user_question).strip()
            user_question = re.sub(r"\s+", " ", user_question).strip()

        # 如果生成的问题为空（例如模板只有一个可选参数且未被选中），则创建一个简单问题
        if not user_question:
            desc = tool_info["definition"]["tool_description"]
            user_question = f"帮我{desc}"

        # --- 5. 组合最终的输入和输出 ---
        input_prompt = BASE_PROMPT.format(
            tool_definition=tool_definition_str,
            user_question=user_question
        )
        
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

# --- Main execution ---
if __name__ == "__main__":
    NUM_SAMPLES = 1000
    generated_data = generate_data(NUM_SAMPLES)

    file_path = "./data/finetuning_data.csv"
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(generated_data)

    print(f"✅ 成功生成了 {NUM_SAMPLES} 条包含用户问题的微调数据，并已保存到文件：{file_path}")
