import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 初始化 FastAPI 应用
app = FastAPI()

# 配置 OpenAI 客户端
os.environ['ARK_API_KEY'] = "2ad632c1-850e-49df-a9ae-3720341c687c"
client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


# 定义请求体模型
class CardTextRequest(BaseModel):
    input_text: str


# 定义响应体模型
class CardInfoResponse(BaseModel):
    year: str
    program: str
    card_set: str
    card_num: str
    athlete: str


# 系统提示词
prompt = """
你是一个专业的体育卡牌数据解析助手。请严格按照以下规则处理输入的卡牌文本，输出JSON格式结果：

1. 字段说明（输出字段必须为以下名称）：
   - year: 年份（取前四位数字，如'2023-24'取2023）
   - program: 卡系列（仅限Aficionado/Prominence/Honors/Playoffs/Kobe Bryant Box Set/Torque/Absolute Memorabilia/Americana I Century Collection/Treble/La Liga/Retail/Collegiate Draft Picks/Prizm UEFA Euro/Hoops Premium Stock/Momentum/Hoops/Leather and Lumber/Premium Stock /Court Kings/Gridiron Kings/Prime Cuts IV/Preferred/Frito Lay/Contenders Draft Picks /Score Premier League/Hogg Heaven/Diamond Kings II/Threads/Dominion/Donruss Elite Serie A/Noir /Select FIFA /Absolute/Americana Heroes and Legends/Cooperstown/Gala/Complete/Vertex/Innovation/Luminance/PhotoGenic/Mosaic Serie A/US National Team/Cooperstown II/Prizm FIFA World Cup Qatar 2022 TM/Black /Americas Pastime/Opulence/Nobility/WNBA Select/Mosaic La Liga and Serie A/The Rookies/Clearly Donruss/Justin Bieber 2/Luxury Suite/Donruss/Obsidian/Photogenic/Zenith/Prizm Draft Picks/Vanguard/Limited Cuts/Americana II/NBA G League/Classics/Century Collection/Recon/Monopoly/Prizm Deca /Base Brand/EuroLeague Donruss /USA Baseball National Team Box Set/Excalibur/Rated Rookies/Intrigue/USA Baseball Champions/Mosaic Series II/Chronicles/Flawless/Crown Royale/Playoffs Bonus Pack/One and One/Signature Series/Timeless Treasures/Americana 2011/Playoff/Mosaic FIFA/Elite La Liga/Impeccable /Rookies and Stars Longevity/Spectra /Beach Boys 50th Anniversary/EuroLeague Donruss/Black Gold/Classics Signatures/Noir/Contenders Draft Picks/Throwback Threads/Country Music/Gridiron/Elite Black Box/Stars and Stripes/Legacy/Prizm Monopoly/EuroLeague Crown Royale /Kobe Eminence 2017/Elite Series/Clear Vision/Elite Extra Edition/Donruss Optic/Black Gold Collegiate/Select Premier League/Flawless Collegiate/One/Contenders /Contenders Patches/Champions/Gridiron Gear/Playbook/Rookies and Stars/WNBA Origins/Certified /Donruss Elite FIFA/Origins/Elite Draft Picks/Inscriptions/Score Serie A/Select Draft Picks/Customer Service Rewards/WNBA Revolution/Mosaic La Liga/Prime Cuts/Americana Sports Legends/Justin Bieber/WNBA Donruss/Marquee/Ascension/Select UEFA Euro Preview/Black/Contenders/Boys of Summer/Phoenix /Eminence FIFA World Cup Qatar 2022 TM/Panini/NBA Season Update/Illusions/Gold Standard/Rookie Anthology/Score Rookie and Traded/Phoenix/Victory Lane/Hall of Fame/Score FIFA/Contenders Optic/Score Halloween/Eminence/Replay/Chronicles Draft Picks/Past and Present/Mosaic Euro/Kit Young Hawaii/Passing the Torch/National Treasures/Golden Age/Unparalleled/Impeccable/Michael Jackson I/Americana/Score/College Sets Multi Sport/Select/Titanium/Donruss Elite Premier League/Caitlin Clark Collection/Score Ligue 1 Uber Eats/Hall of Fame BB/Select FIFA/Luxe/Rookies and Stars /Mosaic/Silver Signatures/Paramount/Prizm Perennial Draft Picks  /Pantheon/Three and Two/Prizm/Grand Reserve/Flux/Immaculate/Mosaic Premier League/NXT/WNBA Prizm/Revolution/Xr/Playoffs Bonus Pack 2/Triple Play/Infinity/Thanksgiving Day Classic/Elements/Diamond Kings I/Studio/Totally Certified/Prestige/Certified Cuts/FIFA Womens World Cup/PhotoGenic /Essentials/Spectra/Hometown Heroes/Absolute Memorabilia Update/Elite/Prime Signatures/Select Serie A/Certified Materials/Base/Immaculate Collegiate/Majestic/Status/Epix/Classics Premium Edition/K League/Cornerstones/Americana Celebrity Cuts/National Treasures Collegiate/Origins /Brilliance/EuroLeague Prizm /One Direction/Prime/Certified/Revolution /Immaculate Collection/Donruss Series II/Encased/Prizm Copa America/Select La Liga/Limited/Donruss Elite/Plates and Patches/Signatures/Chronicles Draft Picks CFB/Crusade/Pinnacle/NFL Playoffs/Capstone/World Cup Prizm/Diamond Kings/）
   - card_set: 卡种（卡系列后的第一个主要特征短语）
   - card_num: 卡编号（以#开头的最早出现的连续字符）
   - athlete: 球员名称（最后出现的人名，需包含姓和名）

2. 处理规则：
   - 字段不存在时设为空字符串
   - 忽略括号内内容和特殊标记（如RC/SP）
   - 保持原始文本顺序，不要重组内容
   - 优先匹配卡系列列表，未匹配到则留空

3. 示例：
输入：Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
输出：{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"#276","athlete":"Bryan Bresee"}
"""


# 定义 FastAPI 接口
@app.post("/parse-card", response_model=CardInfoResponse)
async def parse_card(request: CardTextRequest):
    try:
        # 调用大模型 API
        completion = client.chat.completions.create(
            model="ep-20250206111337-5d7s8",  # 替换为你的模型 endpoint ID
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request.input_text},
            ],
        )

        # 解析大模型返回的 JSON 结果
        result = completion.choices[0].message.content
        return eval(result)  # 将字符串转换为字典返回
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 运行服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
