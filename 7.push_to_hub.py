import os
from huggingface_hub import HfApi, login

def main():
    """
    主函数，引导用户完成模型上传流程。
    """
    # --- 1. 定义本地模型路径 ---
    local_model_path = "./models/merged_gemma_lora"
    
    if not os.path.isdir(local_model_path):
        print(f"❌ 错误：本地模型文件夹不存在于 '{local_model_path}'")
        print("请确保您已经成功运行了模型融合脚本。")
        return

    print("=" * 40)
    print("🚀 开始推送模型到 Hugging Face Hub 🚀")
    print("=" * 40)
    
    # --- 2. 获取用户信息 ---
    hf_username = input("请输入您的 Hugging Face 用户名: ")
    repo_name = input(f"请输入您想创建的仓库名 (例如: {os.path.basename(local_model_path)}): ")
    repo_id = f"{hf_username}/{repo_name}"

    print(f"✅ 准备就绪，将要把 '{local_model_path}' 的内容推送到仓库 '{repo_id}'")
    
    # --- 3. 登录 ---
    print("\n--- 步骤 1/2: 登录 ---")
    print("您需要一个有 'write' 权限的 Hugging Face Access Token。")
    print("请从这里获取: https://huggingface.co/settings/tokens")
    login()
    print("✅ 登录成功！")

    # --- 4. 创建仓库并上传 ---
    api = HfApi()
    try:
        print("\n--- 步骤 2/3: 创建或确认远程仓库 ---")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True  # 如果仓库已存在，则不报错
        )
        print(f"✅ 仓库 '{repo_id}' 已在 Hub 上准备就绪。")

        print("\n--- 步骤 3/3: 上传文件 ---")
        print("这可能需要一些时间，取决于您的网络速度和模型大小...")
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload merged model from {local_model_path}"
        )
        print(f"🎉 成功！模型已上传至: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ 上传过程中发生错误: {e}")
        print("请检查您的网络连接、token权限以及仓库名称是否有效。")

    print("=" * 40)

if __name__ == "__main__":
    main()
