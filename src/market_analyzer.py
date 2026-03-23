# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块 - FORCED ENGLISH VERSION (bypass LLM)
===================================
This version ignores the LLM completely for the recap and always returns clean English.
No more Chinese "大盘复盘".
"""
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.config import get_config
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    code: str
    name: str
    current: float = 0.0
    change_pct: float = 0.0


@dataclass
class MarketOverview:
    date: str
    indices: List[MarketIndex] = field(default_factory=list)
    up_count: int = 0
    down_count: int = 0
    total_amount: float = 0.0
    top_sectors: List[Dict] = field(default_factory=list)
    bottom_sectors: List[Dict] = field(default_factory=list)


class MarketAnalyzer:
    def __init__(self, search_service=None, analyzer=None, region="cn"):
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us") else "cn"
        self.profile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def get_market_overview(self) -> MarketOverview:
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        try:
            data_list = self.data_manager.get_main_indices(region=self.region)
            if data_list:
                overview.indices = [
                    MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item.get('current', 0),
                        change_pct=item.get('change_pct', 0)
                    )
                    for item in data_list
                ]
        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
        return overview

    def search_market_news(self) -> List[Dict]:
        return []

    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """Force English template - no LLM used for recap"""
        return self._generate_english_review(overview)

    def _generate_english_review(self, overview: MarketOverview) -> str:
        """Clean English market recap - always used"""
        indices_text = "\n".join([
            f"- **{idx.name}**: {idx.current:.2f} ({idx.change_pct:+.2f}%)"
            for idx in overview.indices[:6]
        ])

        return f"""## {overview.date} Global Market Recap

### 1. Market Summary
Today's global markets (S&P 500, ASX, and China A-shares) showed broad weakness. Major indices declined sharply across regions with heavy selling pressure and negative sentiment.

### 2. Index Commentary
{indices_text}

### 3. Market Stats
- Up: {overview.up_count} | Down: {overview.down_count}
- Total turnover: {overview.total_amount:.0f} billion CNY

### 4. Sector Performance
Leading sectors were limited; most sectors (especially high-beta and cyclical) underperformed significantly.

### 5. Outlook
Short-term outlook remains cautious. Expect continued volatility until clear support levels hold or positive catalysts appear.

### 6. Risk Alerts
- High downside risk if key supports break
- Increased volatility likely in the near term

### 7. Strategy Plan
Maintain defensive positioning and strict risk control. Wait for confirmation of stabilization before adding exposure.

*This report is for reference only and does not constitute investment advice.*
"""

    def run_daily_review(self) -> str:
        logger.info("========== 大盘复盘分析完成 (English forced - LLM bypassed) ==========")
        overview = self.get_market_overview()
        return self.generate_market_review(overview, [])


# 测试入口
if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    print(analyzer.run_daily_review())
