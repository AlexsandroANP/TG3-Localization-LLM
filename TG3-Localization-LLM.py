import json
import re
from openai import OpenAI
import os
import shutil  
import json_repair



class ModelConfig:
    ####################################
    # 调用云端 LLM，先填写相应的 API-KEY #
    # 调用本地 LLM，注意端口是否正确     #
    ####################################
    system_prompt = '''
    作为一名深谙文学翻译之道的专家，你肩负起了一项极具挑战性的游戏本地化翻译任务。
    该游戏的原始语言为英语，背景设定在欧洲中世纪，
    玩家将沉浸在一个充满探索、经营生意、社交互动的虚拟世界中，全方位体验中世纪的模拟人生。
    游戏开发者将提供两项至关重要的信息：
    一是文本在游戏中的具体定义（<Key>...</Key>），
    二是文本在游戏中的实际显示内容（<Text>...</Text>）。
    你的核心任务是基于文本的定义，将显示文本精准且艺术地翻译成中文。
    译文需具备浓郁的中世纪文学风格，
    使用翻译腔的表达手法，用中文表达出英文的句式结构，
    将中世纪英文的语气、韵味和风格传达给玩家，追求翻译品质，
    要文笔优美，用词典雅。
    通过你的翻译，
    玩家在中文语境下应能无缝感受到游戏所营造的中世纪氛围，
    确保语言转换后的游戏体验依然生动而真实。
    输出格式：
    使用 JSON 模板输出内容
    ```json
        {
        "Key":""
        "Text":""
        }
    ```
    '''
    temperature = 0.65
    max_tokens = 1024*4
    provider = {
        'default': {
            'key': 'your-api-key',
            'endpoint': 'http://localhost:3000/v1/',
            'models': ['qwen', 'llama', 'minicpm']
        },
        'Xinference': {
            'key': 'empty-api-key',
            'endpoint': 'http://192.168.31.170:9997/v1/',
            'models': ['qwen2.5-instruct', 'qwen2-instruct']
        },
        'zhipu': {
            'key': 'zhipu-api-key',
            'endpoint': 'https://open.bigmodel.cn/api/paas/v4/',
            'models': [
                'glm-4-air', 'glm-4', 'glm-4-airx', 'glm-4-flash', 'glm-4-0520', 'glm-4-plus',
                'glm-4-alltools', 'glm-4-assistant', 'glm-4-long', 'codegeex-4', 'glm-3-turbo'
            ]
        },
        'deepseek': {
            'key': 'deepseek-api-key',
            'endpoint': 'https://api.deepseek.com',
            'models': ['deepseek-chat', 'deepseek-coder']
        }
    }



class ModelRequest:
    def __init__(self, ModelConfig, model_name):
        self.model_name     = model_name
        self.prompt         = ModelConfig.system_prompt
        self.temperature    = ModelConfig.temperature
        self.max_tokens     = ModelConfig.max_tokens
        found               = False
        
        for provider_name, provider_info in ModelConfig.provider.items():
            if model_name in provider_info['models']:
                self.api_key        = provider_info['key']
                self.model_endpoint = provider_info['endpoint']
                self.client         = OpenAI(api_key = self.api_key, base_url = self.model_endpoint)
                found               = True

                print(f"\n{Colors.GREEN}{'='*15}{Colors.RESET}")
                print(f"{Colors.BLUE}已找到 {Colors.YELLOW}{model_name}{Colors.BLUE} 模型，由 {Colors.YELLOW}{provider_name}{Colors.BLUE} 提供服务{Colors.RESET}")
                break
        if not found:
            raise ValueError(f"不在模型配置列表中: {model_name}")
        
    def single_round_request(self,user_content):
            client      = self.client

            response = client.chat.completions.create(
                model       = self.model_name,
                messages    = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_content}
                    ],
                temperature = self.temperature,
                max_tokens  = self.max_tokens
            )
            return response.choices[0].message.content
    
    
    def multi_round_request(self, user_content):
        client   = self.client
        # Initialize messages with the system prompt
        messages = [{"role": "system", "content": self.prompt}]
        # Split the user content into chunks that do not exceed max_tokens
        chunks          = []
        current_chunk   = []
        content_chunks  = self.split_content_into_chunks(user_content)
        words           = user_content.split()

        for word in words:
            if len(" ".join(current_chunk + [word])) <= self.max_tokens:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Process each chunk in a separate round
        for chunk in content_chunks:
            messages.append({"role": "user", "content": chunk})

            response = client.chat.completions.create(
                model       = self.model_name,
                messages    = messages,
                temperature = self.temperature,
                max_tokens  = self.max_tokens
            )
            assistant_response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_response})

        # Final request to summarize the entire conversation
        summary_request = "请总结以上对话内容。"
        messages.append({"role": "user", "content": summary_request})

        final_response = client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_tokens
        )

        return final_response.choices[0].message.content



class MaybeFoolJsonalize:
    def __init__(self, input_str):
        self.input_str  = str(input_str)
        self.result     = None
        self.status     = False
        self.parse_json()

    def parse_json(self):
        try:
            self.result = json.loads(self.input_str)
            self.status = True
        except json.JSONDecodeError:
            self.result, self.status = self.fix_object(self.input_str)
            if not self.status:
                self.result, self.status = self.use_json_repair(self.result)
                if not self.status:
                    print('从内容中获取 JSON --> 失败')

    def fix_object(self,input_str):
        # ref: https://www.bigmodel.cn/dev/howuse/jsonformat
        _status     = True
        _pattern    = r"\{(.*)\}"
        _match      = re.search(_pattern, input_str)
        input_str   = "{" + _match.group(1) + "}" if _match else input_str

        # 清理 json 字符串。
        input_str = (
            input_str.replace("{{", "{")
            .replace("}}", "}")
            .replace('"[{', "[{")
            .replace('}]"', "}]")
            .replace("\\", " ")
            .replace("\\n", " ")
            .replace("\n", " ")
            .replace("\r", "")
            .strip()
        )
        # 移除 JSON Markdown 框架
        match = re.search(r'```json\s*(\{.*?\})```', input_str, re.DOTALL)
        if match:
            input_str = match.group(1)
        else:
            input_str = input_str
        try:
            _result = json.loads(input_str)
        except json.JSONDecodeError:
            _result = input_str
            _status = False
    
        return _result,_status

    def use_json_repair(self, input_str):
        _status     = True
        input_str   = str(input_str)
        _json_info  = json_repair.repair_json(json_str=input_str,ensure_ascii=False)
        try:
            _result = json.loads(_json_info)
        except json.JSONDecodeError:
            _result = input_str
            _status = False

        return _result,_status



class Jsonalize:
    def __init__(self, input_str):
        self.input_str = str(input_str)
        self.decoded_object = json_repair.repair_json(json_str         = self.input_str, 
                                                      return_objects   = True,
                                                      ensure_ascii     = False)



class Colors:
    RED         = '\033[91m'
    GREEN       = '\033[92m'
    YELLOW      = '\033[93m'
    BLUE        = '\033[94m'
    MAGENTA     = '\033[95m'
    CYAN        = '\033[96m'
    RESET       = '\033[0m'
    ORANGE      = '\033[38;5;208m'
    PURPLE      = '\033[38;5;165m'
    LIGHT_BLUE  = '\033[38;5;39m'



class LocalizationManager:
    def __init__(self, game_dir):
        self.game_dir           = game_dir
        self.media_dir          = os.path.join(game_dir, 'media')
        self.localization_dir   = os.path.join(self.media_dir, 'localization')
        self.backup_dir         = os.path.join(self.media_dir, 'localization_backup')

    def read_locdirect_file(self, file_name):
        file_path = os.path.join(self.localization_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"{Colors.GREEN}成功读取文件:\n{file_path}{Colors.RESET}")
                return content
        except FileNotFoundError:
            error_message = "文件未找到，请检查文件路径。"
            print(f"{Colors.RED}{error_message}{Colors.RESET}")
        except Exception as e:
            error_message = f"发生错误: {e}"
            print(f"{Colors.RED}{error_message}{Colors.RESET}")

    def backup_localization_folder(self):
        if not os.path.exists(self.backup_dir):
            shutil.copytree(self.localization_dir, self.backup_dir)
            print(f"{Colors.GREEN}已创建备份文件夹: \n{self.backup_dir}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}备份文件夹已存在: \n{self.backup_dir}{Colors.RESET}")

    def write_locdirect_file(self, file_name, content):
        file_path = os.path.join(self.localization_dir, file_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
                print(f"{Colors.GREEN}{'='*20}\n已将修改后的内容写入：\n{file_path}{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.RED}写入文件时发生错误:\n{e}{Colors.RESET}")
    


class TranslationManager:
    def parse_locdirect_data(self, content):
        # 使用正则表达式提取 LocDirectEntry 的 Key 和 Text
        loc_data    = []
        entries     = re.findall(r'LocDirectEntry\s*{([^}]*)}', content, re.DOTALL)
        
        for entry in entries:
            key_match   = re.search(r'Key\s*=\s*"([^"]*)";', entry)
            text_match  = re.search(r'Text\s*=\s*"([^"]*)";', entry)
            if key_match and text_match:
                key     = key_match.group(1)
                text    = text_match.group(1)
                loc_data.append({"Key": key, "Text": text})

        return loc_data

    def translate_and_polish(self, entry, model_name):
        translation_manager         = ModelRequest(ModelConfig, model_name)
        translate_and_polish_text   = translation_manager.single_round_request(str(entry))
        jsonalizer                  = MaybeFoolJsonalize(translate_and_polish_text)
        json_object                 = jsonalizer.result
        json_str                    = json.dumps(jsonalizer.result, indent=4, ensure_ascii=False)
        
        # 使用颜色类来优化输出样式
        print(f'{Colors.GREEN}{"="*20}\n完成本地化{Colors.RESET}\n'
            f'{Colors.BLUE}{"-"*20}\n原文{Colors.RESET}{Colors.YELLOW}{Colors.RESET}\n'
            f'{Colors.CYAN}{entry}{Colors.RESET}'
            f'{Colors.BLUE}\n{"-"*20}\n译文{Colors.RESET}{Colors.YELLOW}{Colors.RESET}'
            f'{Colors.CYAN}{json_str}{Colors.RESET}')
        return json_object



def main(game_dir,file_name,output_file_name,model_name):
    localization_manager = LocalizationManager(game_dir)
    localization_manager.backup_localization_folder()
    content              = localization_manager.read_locdirect_file(file_name)
    translation_manager  = TranslationManager()
    loc_data             = translation_manager.parse_locdirect_data(content)

    # 复制一份 file_name 的原内容，并命名为 output_file_name
    localization_manager.write_locdirect_file(output_file_name, content)

   # 处理每个 LocDirectEntry
    for entry in loc_data:
        # 获取翻译后的 entry,替换 content 中的 Text
        # 将修改后的内容写入新的 .loo 文件
        try:
            translated_entry = translation_manager.translate_and_polish(entry,model_name)
            content          = content.replace(f'"{entry["Text"]}"', f'"{translated_entry["Text"]}"')
            localization_manager.write_locdirect_file(output_file_name, content)
        except:
            pass


if __name__ == "__main__":
    ####################
    # *修改实际游戏目录 #
    # *选择目标语言文件 #
    ####################
    model_name       = 'glm-4-flash'
    game_dir         = 'C:\\Games\\The Guild 3'  # 游戏目录路径
    file_name        = 'locdirect_english.loo'  # 语言文件名
    output_file_name = 'modified_locdirect_english_test.loo'  # 翻译输出文件名

    main(game_dir,file_name,output_file_name,model_name)




                       
                       