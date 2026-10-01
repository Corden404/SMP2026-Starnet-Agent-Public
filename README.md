# SMP 2026 Starnet Agent — IP44

IP44 是 SMP 2026 星网挑战赛的最终参赛实现。本仓库发布冻结提交包的原始代码、配置与智能体提示词，并提供离线验证和打包工具。

## 实现

`workspace/team_submission/starnet_model.py` 提供 `ParticipantSquadModel(host_env, person_list, llm)`。宿主负责提供环境公开接口和 LLM 适配器。

- 在 50 节点场景中，侦察前缀依据已公开拓扑的 50 步扩散影响系数排序。
- 使用扫描回执和游说反馈维护公开状态与响应估计。
- 规划器比较游说、断边和屏蔽动作，并保留受时间限制的 PC2 比较逻辑。
- 统一计数器在环境操作和 LLM 调用达到熔断值前停止。

配置与 7 个提示词位于 `workspace/team_submission/`。该目录的 9 个文件与冻结参赛包逐字节一致；各文件 SHA-256 记录在 `submission_manifest.json`。重新打包会使用统一 ZIP 元数据，因此生成 ZIP 的整体哈希可以与原 ZIP 不同。

## 安装

使用 Python 3.11，在仓库根目录执行：

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
git clone --branch v0.3.19 --depth 1 https://github.com/rgCASS/casevo.git vendor/casevo
python -m pip install --no-deps -e vendor/casevo
~~~

Linux/macOS 创建环境后使用 `source .venv/bin/activate`。若宿主提供比赛 SDK `agent_mesa`，提交实现优先使用它；离线测试使用 CaseVO v0.3.19。

## 离线验证

~~~powershell
python -m unittest discover -s tests -v
~~~

测试使用模拟环境和模拟 LLM，不读取真实密钥，不调用比赛服务器或模型服务。验证范围包括源码身份、打包布局、公开拓扑排序、资源计数和真实 CaseVO 链的控制流程；测试结果不代表官方评分。

## 打包

~~~powershell
python scripts/package_submission.py --output dist/ip44.zip
~~~

脚本先核验 9 个文件的哈希，再生成 ZIP。ZIP 根目录包含 `config.json`、`starnet_model.py` 和 `prompt/`，不包含测试、开发工具或额外文件。输出路径已存在时停止，避免覆盖既有包。

## 仓库范围

默认分支的最新快照仅包含 IP44 发布所需内容。清理后的开发提交保留了早期代码演进；历史实验工具依赖的本地实验数据不在本仓库，历史快照不作为当前运行入口。当前支持的验证入口是上述离线测试。

CaseVO 为外部依赖，源码未捆绑到提交包；来源与许可证文本见 `THIRD_PARTY_NOTICES.txt` 和 `licenses/CaseVO-LICENSE.txt`。
