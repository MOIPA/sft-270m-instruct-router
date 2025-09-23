import json
import random
import csv
import re

# --- 1. 新的系统提示词 ---
BASE_PROMPT = """你是一个强大的多模态AI助手。你的核心任务是理解并响应用户的需求。

请遵循以下优先级处理用户输入：
1.  **当用户提供了图片时**：你的首要任务是用自然语言描述图片内容或回答相关问题。**除非用户的文字指令明确要求使用工具**，否则不应调用工具。
2.  **当没有图片，或用户明确要求执行工具操作时**：判断用户的意图是否与下面列出的某个工具有明确匹配。如果匹配，请生成一个用于调用工具的JSON对象。
3.  **所有其他情况**：对于普通对话、问候、开玩笑或任何与工具功能无关的请求，请直接用自然语言回复。

**工具调用规则**：
当你决定调用工具时，必须严格按照MCP协议输出一个JSON对象，不要输出思考过程，也绝对不允许在JSON前后添加任何多余的文字。

**核心规则：处理前置条件**
某些工具的描述中包含了 `[前置条件: ...]`。在调用这些工具之前，你**必须**首先确保它的前置条件已经被满足。如果前置条件尚未满足（例如，你还不知道今天的日期），你的**唯一任务**就是去调用那个能满足前置条件的工具（例如 `get_current_date`）。**绝对不能**在一次输出中，同时输出多个工具调用。

**核心规则：判断任务完成**
当工具的返回结果 `[tool_result]` 中包含了明确的成功标识，如 "success", "成功", "已创建", "已完成", "已打开页面" 等关键词时，这代表你的任务已经完成。此时，你**必须**停止调用任何工具，并基于这个成功的结果，为用户生成一个最终的、确认性的自然语言回答。**绝对不能**再次调用相同的或其他的工具。

**核心规则：处理日期**
你没有任何关于当前日期的先验知识。当用户的请求中包含“今天”、“明天”、“下周三”这样的相对时间描述时，它们本身并不是一个可用的日期。你**必须**首先调用 `get_current_date` 工具来获取当前的绝对日期，然后基于这个结果计算出用户所指的目标日期，才能调用其他工具。

---
# 示例 1: 简单工具调用
[conversation_history]
用户: "帮我查一下2025年9月10号有什么安排"
[your_turn]
{{"tool_name": "get_calendar_events", "arguments": {{"date": "2025-09-10"}}}} 

# 示例 2: 无需调用工具 (普通对话)
[conversation_history]
用户: "你好"
[your_turn]
你好！有什么可以帮你的吗？

# 示例 3: 链式调用的第一步 (信息不足，需获取日期)
[conversation_history]
用户: "帮我在明天下午2点安排一个讨论需求的会议，大约持续一小时"
[your_turn]
{{"tool_name": "get_current_date", "arguments": {{}}}}

# 示例 4: 链式调用的第二步 (已获取日期，继续执行任务)
[conversation_history]
用户: "帮我在明天下午2点安排一个讨论需求的会议，大约持续一小时"
[tool_result]
get_current_date() -> "2025-08-07"
[/tool_result]
[your_turn]
{{"tool_name": "create_calendar_event_api", "arguments": {{"title": "讨论需求的会议", "start_time": "2025-08-08 14:00:00", "end_time": "2025-08-08 15:00:00"}}}}

# 示例 5: 任务完成
[conversation_history]
用户: "帮我在明天下午2点安排一个讨论需求的会议，大约持续一小时"
[tool_result]
get_current_date() -> "2025-08-07"
[/tool_result]
[tool_code]
{{"tool_name": "create_calendar_event_api", "arguments": {{"title": "讨论需求的会议", "start_time": "2025-08-08 14:00:00", "end_time": "2025-08-08 15:00:00"}}}}
[/tool_code]
[tool_result]
create_calendar_event_api() -> "{{\"status\": \"success\", \"message\": \"日程已创建\"}}"
[/tool_result]
[your_turn]
好的，会议已经为您安排在明天下午2点。
---
可用的工具列表如下:
{tool_definition}

---
prompt:<|im_start|>user
"{user_question}"
<|im_end|>
<|im_start|>assistant' 
"""

# --- 2. 根据新的工具定义更新 ---
TOOLS = {
    "clear_all_cache": {
        "definition": {
            "tool_name": "clear_all_cache",
            "tool_description": "清理应用的所有缓存，包括系统缓存、消息缓存和小程序缓存。",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "帮我清理所有缓存",
            "清理一下应用的全部缓存",
            "把所有缓存都删掉"
        ]
    },
    "clear_message_cache": {
        "definition": {
            "tool_name": "clear_message_cache",
            "tool_description": "清理消息模块的所有缓存，主要是图片、视频、文件等聊天附件。",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "清理一下消息缓存",
            "删除聊天图片和文件",
            "消息的缓存怎么清理？"
        ]
    },
    "clear_miniprogram_cache": {
        "definition": {
            "tool_name": "clear_miniprogram_cache",
            "tool_description": "清理所有已安装的小程序的缓存文件。",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "把小程序缓存清一下",
            "删除小程序占用的空间",
            "如何清理小程序缓存？"
        ]
    },
    "create_calendar_event_api": {
        "definition": {
            "tool_name": "create_calendar_event_api",
            "tool_description": "通过API静默创建一个新的日历事件。[前置条件: 如果用户没有提供具体日期，必须先通过 get_current_date 工具获得今天的日期]。如果用户的描述中包含‘今天’、‘明天’等相对时间，你必须先调用 get_current_date 来解析成绝对日期，然后计算出 start_time 和 end_time。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "title": { "type": "string", "description": "The title or subject of the event." },
                    "start_time": { "type": "string", "description": "The event start time in 'YYYY-MM-DD HH:mm:ss' format." },
                    "end_time": { "type": "string", "description": "The event end time in 'YYYY-MM-DD HH:mm:ss' format." }
                },
                "required": ["title", "start_time", "end_time"]
            }
        },
        "arguments_samples": {
            "title": ["项目周会", "和李总开会", "需求评审"],
            "start_time": ["2025-09-12 10:00:00", "2025-09-15 14:30:00"],
            "end_time": ["2025-09-12 11:00:00", "2025-09-15 15:00:00"]
        },
        "question_templates": [
            "帮我安排一个会议，主题是{title}，从{start_time}到{end_time}",
            "创建一个日程：{title}，时间是{start_time}到{end_time}"
        ]
    },
    "create_calendar_event": {
        "definition": {
            "tool_name": "create_calendar_event",
            "tool_description": "创建一个新的日历事件、会议或待办事项。注意：如果用户没有提供具体日期，你应该先调用 get_current_date 工具来获取今天的日期。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "title": { "type": "string", "description": "title or theme of the event/meeting" },
                    "start_time": { "type": "string", "description": "start time of the event。if the user does not provide then ignore this parameter because the create calendar menu will let user choose the time info" }
                },
                "required": ["title"]
            }
        },
        "arguments_samples": {
            "title": ["团队建设", "生日聚餐", "看医生"],
            "start_time": ["明天下午", "周五晚上", "今天中午12点", "null"]
        },
        "question_templates": [
            "帮我创建一个日程，主题是{title}，时间是{start_time}",
            "记一下，{start_time}要{title}",
            "安排一个会议，叫{title}"
        ]
    },
    "decrease_font_size": {
        "definition": {
            "tool_name": "decrease_font_size",
            "tool_description": "减小字体大小一号",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "把字体调小一点",
            "减小字号",
            "字体太大了，缩小一些"
        ]
    },
    "get_calendar_events": {
        "definition": {
            "tool_name": "get_calendar_events",
            "tool_description": "查询指定日期的日程列表。[前置条件: 如果用户没有提供具体日期，必须先通过 get_current_date 工具获得今天的日期]。如果用户的描述中包含‘今天’、‘明天’等相对时间，你必须先调用 get_current_date 来解析成绝对日期。",
            "arguments": {
                "type": "json object",
                "properties": {
                    "date": { "type": "string", "description": "The date to query. 格式是 yyyy-MM-dd" }
                },
                "required": ["date"]
            }
        },
        "arguments_samples": {
            "date": ["2025-09-11", "2025-09-12", "明天", "今天"]
        },
        "question_templates": [
            "查一下{date}的日程",
            "我{date}有什么安排？",
            "看看{date}的日历"
        ]
    },
    "get_current_date": {
        "definition": {
            "tool_name": "get_current_date",
            "tool_description": "获取今天的绝对日期，格式为 YYYY-MM-DD。这是解析'今天'、'明天'等相对时间描述的基础工具，必须在处理任何与日期相关的任务前首先调用。",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "今天几号？",
            "今天是哪一天？",
            "获取当前日期"
        ]
    },
    "increase_font_size": {
        "definition": {
            "tool_name": "increase_font_size",
            "tool_description": "增大字体大小一号",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "把字体调大一点",
            "增大字号",
            "字体太小了，放大一些"
        ]
    },
    "search_contact": {
        "definition": {
            "tool_name": "search_contact",
            "tool_description": "Searches for contacts based on a keyword (such as name, pinyin, or employee ID).",
            "arguments": {
                "type": "json object",
                "properties": {
                    "keyword": { "type": "string", "description": "The keyword to search for, e.g., 'John Doe' or 'zhangsan'." }
                },
                "required": ["keyword"]
            }
        },
        "arguments_samples": {
            "keyword": ["张三", "Li Si", "wangwu", "12345", "项目经理"]
        },
        "question_templates": [
            "帮我找一下{keyword}",
            "查一下联系人{keyword}",
            "搜索{keyword}的联系方式"
        ]
    },
    "set_font_size": {
        "definition": {
            "tool_name": "set_font_size",
            "tool_description": "打开字体大小设置页面",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "打开字体大小设置",
            "我想调整字体大小",
            "在哪里可以设置字体？"
        ]
    },
    "upload_log": {
        "definition": {
            "tool_name": "upload_log",
            "tool_description": "Opens the log upload page, allowing the user to upload application logs.",
            "arguments": {}
        },
        "arguments_samples": {},
        "question_templates": [
            "帮我上传一下日志",
            "我要上报问题，需要上传日志",
            "打开日志上传页面"
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
        # 确保所有必填参数都被填充
        required_args = tool_info["definition"].get("arguments", {}).get("required", [])
        for arg_name in required_args:
            if arg_name in tool_info["arguments_samples"]:
                value = random.choice(tool_info["arguments_samples"].get(arg_name, [None]))
                if value is not None:
                    args[arg_name] = value

        # 随机填充可选参数
        for arg_name, samples in tool_info.get("arguments_samples", {}).items():
            if arg_name not in args: # 如果还未被填充
                # 70%的概率填充可选参数
                if random.random() < 0.7:
                    value = random.choice(samples)
                    if value is not None:
                        args[arg_name] = value

        # 如果这个工具有参数，但最终args还是空的，强行选一个
        if not args and tool_info.get("arguments_samples"):
            arg_name = random.choice(list(tool_info["arguments_samples"].keys()))
            value = random.choice(tool_info["arguments_samples"].get(arg_name, [None]))
            if value is not None:
                args[arg_name] = value


        # --- 4. 根据参数和模板生成用户问题 ---
        user_question = ""
        if "question_templates" in tool_info and tool_info["question_templates"]:
            question_template = random.choice(tool_info["question_templates"])

            # 准备格式化字典，对于缺失的参数用空字符串代替
            format_args = {key: "" for key in tool_info.get("arguments_samples", {})}
            format_args.update(args)

            try:
                user_question = question_template.format(**format_args)
                # 清理因为缺失参数可能导致的语法问题
                user_question = user_question.replace("，时间是", "").replace("时间是", "").strip()
                user_question = re.sub(r'[\s，]+$', '', user_question) # 移除末尾的空格和逗号
                user_question = re.sub(r'^\s*要', '', user_question) # 移除开头的"要"
            except KeyError as e:
                # 如果模板中的某个key不存在于format_args中，跳过这个模板
                # print(f"Skipping template due to KeyError: {e}. Template: '{question_template}'")
                continue


        # 如果生成的问题为空，则创建一个简单问题
        if not user_question:
            desc = tool_info["definition"]["tool_description"].split("。")[0] # 取第一句描述
            user_question = f"帮我{desc}"


        # --- 5. 组合最终的输入和输出 ---
        # 对于需要获取日期的工具，模拟链式调用场景
        is_date_dependent = "date" in tool_info["definition"].get("arguments", {}).get("properties", {})
        use_relative_date = random.random() < 0.5 and "date" in args and args["date"] in ["今天", "明天"]

        if is_date_dependent and use_relative_date:
            # 场景1: 链式调用的第一步，模型应该去获取日期
            input_prompt = BASE_PROMPT.format(
                tool_definition=json.dumps(TOOLS["get_current_date"]["definition"], ensure_ascii=False, indent=2),
                user_question=user_question
            )
            output_obj = {
                "tool_name": "get_current_date",
                "arguments": {}
            }
        else:
            # 场景2: 普通的单步调用
            input_prompt = BASE_PROMPT.format(
                tool_definition=tool_definition_str,
                user_question=user_question
            )
            output_obj = {
                "tool_name": tool_name,
                "arguments": args
            }

        output_str = "output：" + json.dumps(output_obj, ensure_ascii=False)

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
