# KOL Connect [产品文档](https://f323gpxjw1d.larksuite.com/docx/P9Wbd8szloHAWKxL9vkuqbJBssF?from=from_copylink)


## 1. 业务目标

- 关键词：精准投放；增长规划；降本增效

- 辅助营销负责人根据营销决策制定投放方案
- 帮助精准定位目标 KOL，并给出投放方案
- 根据投放方案，下发执行方案到 150/天 的投放实习生
- 分析投放方案增长效果，反馈此前制定的 KOL 投放策略，优化红人投放逻辑


## 2. 业务流程

- Marketing 负责人决策
    - 投放产品类型/关键词
    - 预算
    - 增长目标
    - 投放节奏

- 根据方案给出红人建联规划
    - 输入搜索关键词
        - 输入搜索关键词：AI Tool
    - KOL 内容解析
        - 整理返回的视频内容进行 KOL 属性归纳，填充到 KOL 画像
        - 对 KOL 画像标签 embbeding，服务后续检索召回
    - KOL 召回
        - 输入 Prompt = 营销决策，GPT 从 embbeding 库内搜索 KOL 列表
    - KOL 重排序
        - 粉丝活跃度：观看量/粉丝数
        - 粉丝互动率：评论量/粉丝数
        - 判断渠道内容定位和产品定位是否足够高
        - 判断渠道渠道订阅量和视频平均观看量差距是否足够大，订阅几十万用户，平均观看量只有 1K 左右，谨慎合作
        - 判断单个视频内容是否有较多评论、点赞，多多益善，并且从中整理出检索关键词，embbeding 嵌入

- KOL 建联
    - 根据 KOL 画像，写出建联冷启动文案和提示，进入 KOL 建联模式
    - 投放实习生进入建联模式开启建联
        - 不依赖实习生强投放产品类型背景
        - 不依赖高学习成本
        - 不依赖强管理和引导成本
        - 实习生上手即用


## 3. Demo

- step by step 如下：

交互流程: 生产投放决策 -> 搜索投放关键词 -> KOL 内容实习解析 -> KOL 召回 -> KOL 重排序 -> KOL 建联

# KOL 画像建模
## 内容画像

- 互动率：评论量/粉丝数
- 活跃度：观看量/粉丝数
- 好评率：点赞数/观看量
- 讨论度：评论数/观看量
- 正评率：正面评论量/评论量
- 负评率：负面评论量/评论量

## 受众画像

- 年龄
- 性别
- 区域
- 粉丝可信度
- 观看时段
- 相似受众网红


```python
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional,Tuple
```


```python
# YOUTUBE API Setting
YOUTUBE_DEVELOPER_KEY = 'GOOGLE YOUTUBE DEVELOPER KEY'
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# OpenAI API Setting
openAIConfig = {'api_key': 'OPENAI API KEY'}
openai_client = OpenAI(**openAIConfig)
```


```python
@dataclass
class KOLAnalysis:
    interaction_rate: float = 0.0 # 互动率
    activity_rate: float = 0.0 # 活跃度
    like_rate: float = 0.0 # 好评率
    discussion_rate: float = 0.0 # 讨论度
    content_quality: bool = False # 内容质量
    subscriber_view_gap: bool = False # 订阅用户质量
    similarity_score: float = 0.0 # embbeding 嵌入检索时，相似性分数存储

@dataclass
class CommentAnalysis:
    sentiment: str = ""
    keywords: str = ""

@dataclass
class KOLInfo:
    channel_title: str = ""
    subscriber_count: int = 0
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    video_id: str = ""
    channel_id: str = ""
    title: str = ""
    analysis: KOLAnalysis = field(default_factory=KOLAnalysis)
    comment_analysis: CommentAnalysis = field(default_factory=CommentAnalysis)
    
    def __str__(self):
        engagement_rate = (self.like_count + self.comment_count) / self.view_count if self.view_count > 0 else 0
        return (f"KOL: {self.channel_title}\n"
                f"Title: {self.title}\n"
                f"Channel ID: {self.channel_id}\n"
                f"Subscribers: {self.subscriber_count}\n"
                f"Average Views: {self.view_count}\n"
                f"Engagement Rate: {engagement_rate:.2%}\n"
                f"Interaction Rate: {self.analysis.interaction_rate:.2%}\n"
                f"Activity Rate: {self.analysis.activity_rate:.2%}\n"
                f"Sentiment: {self.comment_analysis.sentiment}\n"
                f"Keywords: {self.comment_analysis.keywords}\n"
                f"Similarity Score: {self.analysis.similarity_score:.4f}\n")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KOLInfo':
        analysis_data = data.get('analysis', {})
        comment_analysis_data = data.get('comment_analysis', {})
        
        return cls(
            channel_title=data.get('channel_title', ""),
            subscriber_count=data.get('subscriber_count', 0),
            view_count=data.get('view_count', 0),
            like_count=data.get('like_count', 0),
            comment_count=data.get('comment_count', 0),
            video_id=data.get('video_id', ""),
            channel_id=data.get('channel_id', ""),
            title=data.get('title', ""),
            analysis=KOLAnalysis(
                interaction_rate=analysis_data.get('interaction_rate', 0.0),
                activity_rate=analysis_data.get('activity_rate', 0.0),
                like_rate=analysis_data.get('like_rate', 0.0),
                discussion_rate=analysis_data.get('discussion_rate', 0.0),
                content_quality=analysis_data.get('content_quality', False),
                subscriber_view_gap=analysis_data.get('subscriber_view_gap', False)
            ),
            comment_analysis=CommentAnalysis(
                sentiment=comment_analysis_data.get('sentiment', ""),
                keywords=comment_analysis_data.get('keywords', "")
            )
        )

def convert_to_kol_info_list(kol_data: List[Dict[str, Any]]) -> List[KOLInfo]:
    return [KOLInfo.from_dict(data) for data in kol_data]
```


```python
# KOL 内容解析
def youtube_search(query: str, max_results: int = 10) -> Optional[List[KOLInfo]]:
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_DEVELOPER_KEY)
    try:
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=max_results
        ).execute()
        
        kol_info_list = []
        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                video_id = search_result["id"]["videoId"]
                video_response = youtube.videos().list(
                    part="statistics,snippet",
                    id=video_id
                ).execute()
                video_data = video_response["items"][0]
                channel_id = video_data["snippet"]["channelId"]
                
                channel_response = youtube.channels().list(
                    part="statistics",
                    id=channel_id
                ).execute()
                channel_data = channel_response["items"][0]
                
                kol_info = KOLInfo(
                    title=search_result["snippet"]["title"],
                    channel_title=search_result["snippet"]["channelTitle"],
                    view_count=int(video_data["statistics"]["viewCount"]),
                    like_count=int(video_data["statistics"].get("likeCount", 0)),
                    comment_count=int(video_data["statistics"].get("commentCount", 0)),
                    subscriber_count=int(channel_data["statistics"]["subscriberCount"]),
                    video_id=video_id,
                    channel_id=channel_id
                )
                
                # 这里可以添加 analyze_kol 和 analyze_comments 的调用
                # kol_info.analysis = analyze_kol(kol_info)
                # kol_info.comment_analysis = analyze_comments(kol_info.video_id)
                
                kol_info_list.append(kol_info)
        
        return kol_info_list
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None
```


```python
def analyze_kol(video: KOLInfo) -> KOLAnalysis:
    subscriber_count = video.subscriber_count
    view_count = video.view_count
    like_count = video.like_count
    comment_count = video.comment_count

    return KOLAnalysis(
        interaction_rate=comment_count / view_count if view_count > 0 else 0,
        activity_rate=view_count / subscriber_count if subscriber_count > 0 else 0,
        like_rate=like_count / view_count if view_count > 0 else 0,
        discussion_rate=comment_count / view_count if view_count > 0 else 0,
        content_quality=like_count > 100,  # 简单示例
        subscriber_view_gap=subscriber_count > 10 * view_count
    )
```


```python
# 第三方数据源拿到红人合作费用预估
# TOOD:
```


```python
def analyze_comments(video_id: str) -> Optional[CommentAnalysis]:
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_DEVELOPER_KEY)
    try:
        comments_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()

        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] 
                    for item in comments_response["items"]]

        # 使用OpenAI API进行情感分析
        sentiment_analysis = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze the sentiment of these comments."},
                {"role": "user", "content": f"{comments}"}
            ]
        )

        # 提取关键词
        keywords_analysis = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract keywords from these comments."},
                {"role": "user", "content": f"{comments}"}
            ]
        )

        return CommentAnalysis(
            sentiment=sentiment_analysis.choices[0].message.content.strip(),
            keywords=keywords_analysis.choices[0].message.content.strip()
        )

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None
```


```python
def get_embeddings(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    return openai_client.embeddings.create(input=text, model=model).data[0].embedding
```


```python
def recall_kols(kol_data: List[KOLInfo], kol_profile_requirements: str, top_k: int = 10) -> Tuple[List[int], List[float]]:
    profile_embedding = get_embeddings(kol_profile_requirements)
    kol_embeddings = np.array([get_embeddings(str(kol)) for kol in kol_data])
    
    similarities = cosine_similarity([profile_embedding], kol_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices.tolist(), similarities[top_indices].tolist()
```


```python
def analyze_kols_with_cot(recalled_kols: List[KOLInfo], marketing_strategy: str) -> str:
    kol_info_str = "\n".join(str(kol) for kol in recalled_kols)
    cot_prompt = f"""
    Given the following marketing cot prompt:
    {marketing_strategy}
    
    And the following list of potential KOLs:
    {kol_info_str}
    
    Please provide your analysis and final recommendations.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a marketing expert specializing in influencer marketing for tech products."},
            {"role": "user", "content": cot_prompt}
        ]
    )
    
    return response.choices[0].message.content
```


```python
def get_kol_data(query: str) -> List[KOLInfo]:
    kol_infos = youtube_search(query)
    kol_data = []
    for kol_info in kol_infos:
        # 假设 video 已经是 KOLInfo 对象
        analysis = analyze_kol(kol_info)
        comment_analysis = analyze_comments(kol_info.video_id)
        
        # 更新 KOLInfo 对象的 analysis 和 comment_analysis 属性
        kol_info.analysis = analysis
        kol_info.comment_analysis = comment_analysis
        
        kol_data.append(kol_info)
    
    return kol_data
```


```python
def find_best_kols(kol_data: List[KOLInfo], kol_profile_requirements: str, marketing_strategy_prompt: str) -> str:
    # Recall phase
    recalled_indices, similarities = recall_kols(kol_data, kol_profile_requirements)
    
    # Prepare recalled KOLs data for CoT analysis
    recalled_kols = [kol_data[i] for i in recalled_indices]
    
    # Add similarity scores to recalled KOLs
    for kol, similarity in zip(recalled_kols, similarities):
        kol.analysis.similarity_score = similarity
    
    # CoT analysis
    cot_analysis = analyze_kols_with_cot(recalled_kols, marketing_strategy_prompt)
    
    return cot_analysis
```


```python
# 内容解析
# query target videos
query = "AI Tools"
kol_data = get_kol_data(query)
```


```python
# 召回 & 重排序
# KOL Profile Prompt => 用于 备选 KOL 召回
kol_profile_requirements = """
    AI Tool产品的KOL，内容关注科技和互联网工具，受众为科技爱好者和专业人士
"""

# Martketing Strategy => 用户从召回的 KOL 中，使用 CoT 挑选出符合营销策略要求的 KOL
marketing_strategy_prompt = cot_prompt = f"""
    让我们一步步分析这个问题，每一步都输出分析结果，并将符合条件的KOL作为下一步分析的输入：

    1. 分析每个KOL的内容和受众与我们的AI工具产品的契合度：
    - 详细分析每个KOL的内容主题、风格和目标受众
    - 评估这些因素与我们的AI工具产品的相关性
    - 输出符合契合度标准的KOL列表

    2. 考虑每个符合契合度的KOL的影响力和参与率，评估其对达成10,000次曝光目标的潜力：
    - 分析每个KOL的订阅者数量、平均观看量和互动率
    - 估算每个KOL可能带来的曝光次数
    - 输出最有可能帮助达成曝光目标的KOL列表

    3. 评估如何将1000美元预算有效分配给至少3个KOL：
    - 考虑每个KOL的影响力、参与率和预期曝光效果
    - 提出几种可能的预算分配方案
    - 分析每种方案的优缺点
    - 选择最优的预算分配方案

    4. 最终推荐：
    - 根据前面的分析，选择最符合要求的前3名KOL
    - 对每个选中的KOL，详细说明：
        a. 选择理由
        b. 分配的预算金额
        c. 预期的曝光效果
        d. KOL的YouTube频道URL（格式：https://www.youtube.com/channel/[channel_id]）

    请以中文输出详细的分析过程和最终推荐。确保每一步的推理过程清晰可见，并提供具体数据支持你的结论。
    """

# find best kol by marketing strategy
best_kol_indices = find_best_kols(kol_data, kol_profile_requirements, marketing_strategy_prompt)

print(best_kol_indices)
```

    根据提供的KOL列表和相似度分数，我们将按照您的营销策略进行逐步分析并最终给出推荐。首先，我们从第一步开始分析每个KOL的内容与您的AI工具产品的契合度。
    
    ### 步骤 1: 契合度分析
    1. **Website Learners**
       - **内容主题、风格和目标受众分析**：内容涵盖AI工具，包括视频、图片、音频的生成，以及内容创作、电子邮件自动化等。
       - **相关性评估**：与AI工具产品紧密相关，包括视频制作等。
       - **契合度**：高，与产品内容高度吻合。
    
    2. **Hayls World**
       - **内容主题、风格和目标受众分析**：涉及AI工具、视频编辑等。
       - **相关性评估**：涉及使用AI工具，与产品相关。
       - **契合度**：高，与产品内容一致。
    
    3. **LKLogic**
       - **内容主题、风格和目标受众分析**：AI工具、AI代理等。
       - **相关性评估**：包括AI工具和内容生成，契合度较高。
       - **契合度**：中等，内容与产品关联性较强。
    
    4. **其他KOLs**：
       - 其他KOLs内容也涵盖AI工具，但与产品相关性不及前三个KOLs。
    
    ### 步骤 2: 潜力评估
    基于每个KOL的订阅者数量、平均观看量和互动率，我们评估对于达成10,000次曝光目标的潜力。
    
    - **Website Learners**：
      - **潜力评估**：高，订阅者众多，平均观看量也不俗，互动率合理。
    - **Hayls World**：
      - **潜力评估**：中等，观看量较高，互动率适中。
    - **LKLogic**：
      - **潜力评估**：高，订阅者多，观看量巨大，互动率较低。
    
    ### 步骤 3: 预算分配方案
    在预算分配时，考虑KOL的影响力、参与率和预期曝光效果，提出几种预算分配方案。
    
    1. **方案1**：
       - Website Learners: $500
       - Hayls World: $300
       - LKLogic: $200
       - **优点**：着重投入高影响力KOL，以确保更多曝光。
       - **缺点**：风险集中在一个KOL上。
      
    2. **方案2**：
       - Website Learners: $400
       - Hayls World: $300
       - LKLogic: $300
       - **优点**：平均分配预算，降低风险。
       - **缺点**：某些KOL可能未充分发挥影响力。
    
    ### 最终推荐：
    基于以上分析，最终推荐的前三名KOL如下：
    1. **Website Learners**:
       - **选择理由**：内容与产品高度契合，受众广泛。
       - **分配的预算金额**：$400
       - **预期曝光效果**：120,989 x 4 = 483,956
       - **YouTube频道URL**：[Website Learners](https://www.youtube.com/channel/UCpWT_QfKk7BJIpn709YgsYA)
    
    2. **Hayls World**:
       - **选择理由**：内容相关性较高，有观看量支持。
       - **分配的预算金额**：$300
       - **预期曝光效果**：180,374 x 3.77% = 6,803
       - **YouTube频道URL**：[Hayls World](https://www.youtube.com/channel/UCIxLxlan8q9WA7sjuq6LdTQ)
    
    3. **LKLogic**:
       - **选择理由**：订阅者众多，观看量极高。
       - **分配的预算金额**：$300
       - **预期曝光效果**：1,478,640 x 6.65% = 98,405
       - **YouTube频道URL**：[LKLogic](https://www.youtube.com/channel/UCPkctgt1mTeJWTGj4tq4dPQ)
    
    通过以上分析和推荐，您可以更明智地选择适合您AI工具产品的KOL，并有效分配预算以达到预期曝光目标。希望这些信息对您的营销策略有所帮助！



```python

```
