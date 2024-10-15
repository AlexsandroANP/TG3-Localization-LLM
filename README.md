# 行会3 (The Guild3) 游戏本地化翻译工具

使用大语言模型，对《行会3》游戏文本进行自动本地化翻译，将任意语言翻译为有特定风格和趣味性的中文。

## 功能特点
- **统一接口**：通过包装 OpenAI SDK 的函数，只需模型名称，即可在不同的供应商之间无缝切换，选择不同的模型进行工作。
- **多供应商支持**：支持多个模型供应商，包括 Xinference、zhipu 和 deepseek 等，可以根据模型性能和成本进行选择。
- **丰富的模型选项**：每个供应商提供多种模型选项，如 Qwen、Llama、MiniCPM、InternLM、GLM 等，满足不同翻译风格和需求。
- **动态模型配置**：通过 ModelConfig 类配置不同供应商的API密钥、端点地址和模型列表，ModelRequest 类根据指定模型名称动态选择合适的供应商和模型进行请求。
- **精准翻译**：基于文本定义和显示文本，确保翻译结果的准确性。
- **JSON格式输出**：使用 json-repair 修复生成内容，以标准 JSON 格式输出，便于后续数据处理和集成。
- **安全备份**：自动备份原始本地化文件，防止数据丢失，确保过程安全。


## 配置
- 安装Python环境：确保系统已安装Python 3.x版本。
- 安装依赖库：运行以下命令安装所需库：`openai`, `json_repair`

### ModelConfig 类配置
ModelConfig 类是本项目中的核心配置类，用于管理和配置AI模型的相关信息。通过该类，可以灵活地选择和使用不同的AI模型供应商及其提供的模型。

以下是对 ModelConfig 类的详细说明和使用指南。

1. **system_prompt**：
  - **描述**：系统提示语，用于指导AI模型进行翻译任务。它详细描述了翻译任务的背景、要求和输出格式。
  - **使用**：根据具体需求修改此提示语，以更好地指导AI模型进行翻译。

2. **temperature**：
  - **描述**：控制模型生成文本的随机性。值越小，生成的文本越确定；值越大，生成的文本越多样。
  - **使用**：根据需要调整此参数，以获得更符合预期的翻译结果

3. **max_tokens**：
  - **描述**：控制模型生成文本的最大长度。
  - **使用**：根据需要调整此参数，以适应不同长度的翻译任务。

4. **provider**：
  - **描述**：一个字典，包含多个AI模型供应商的配置信息。每个供应商的配置包括API密钥、端点地址和提供的模型列表。
  - **使用**：根据需要增加新的供应商配置，包括API密钥、端点地址和提供的模型列表。

### main 主程序
- **game_dir**：游戏安装目录的路径。
- **file_name**：需要翻译的本地化文件名。
- **output_file_name**：翻译完成后输出的文件名。
- **model_name**：指定使用的AI模型名称。

## 注意事项
- API密钥和端点地址：确保每个供应商的API密钥和端点地址正确配置，否则会导致模型请求失败。
- 模型名称：在 ModelRequest 中指定的模型名称，必须存在于 ModelConfig 的某个供应商的模型列表中，如果不存在，则需要按照模板，手动添加对用内容。
