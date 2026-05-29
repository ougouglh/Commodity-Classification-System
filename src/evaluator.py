"""
评估模块
用于评估分类系统的性能指标
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class ClassificationEvaluator:
    """分类评估器"""

    def __init__(self):
        self.results = []
        self.errors = []

    def evaluate_predictions(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        level: int = 4
    ) -> Dict:
        """
        评估预测结果

        Args:
            predictions: 预测结果列表
            ground_truth: 真实标签列表
            level: 评估的层级 (1-4)

        Returns:
            评估指标字典
        """
        level_names = ['level_1', 'level_2', 'level_3', 'level_4']
        target_level = level_names[level - 1]

        # 收集结果
        correct = 0
        wrong = 0
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, truth in zip(predictions, ground_truth):
            pred_category = pred.get(target_level, '')
            truth_category = truth.get(target_level, '')

            # 统计
            is_correct = pred_category == truth_category
            if is_correct:
                correct += 1
                category_stats[truth_category]['correct'] += 1
            else:
                wrong += 1
                self.errors.append({
                    'product': truth.get('商品名称', ''),
                    'predicted': pred_category,
                    'ground_truth': truth_category,
                    'confidence': pred.get('confidence', 0),
                    'path': ' > '.join([pred.get(f'level_{i}', '') for i in range(1, 5)])
                })

            category_stats[truth_category]['total'] += 1

        # 计算指标
        total = correct + wrong
        accuracy = correct / total if total > 0 else 0

        # 计算每个分类的准确率
        category_accuracy = {}
        for cat, stats in category_stats.items():
            if stats['total'] > 0:
                category_accuracy[cat] = stats['correct'] / stats['total']

        # 按准确率排序
        sorted_categories = sorted(
            category_accuracy.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'total': total,
            'correct': correct,
            'wrong': wrong,
            'accuracy': accuracy,
            'category_accuracy': dict(sorted_categories),
            'level': level
        }

    def generate_report(self, metrics: Dict) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 60)
        report.append("分类系统评估报告")
        report.append("=" * 60)
        report.append(f"\n总体准确率 (Level {metrics['level']}):")
        report.append(f"  正确: {metrics['correct']}")
        report.append(f"  错误: {metrics['wrong']}")
        report.append(f"  准确率: {metrics['accuracy']:.2%}")
        report.append(f"  总样本: {metrics['total']}")

        # 分类级别统计
        report.append(f"\n各分类准确率 (Top 10 / Bottom 10):")

        sorted_cats = list(metrics['category_accuracy'].items())

        if sorted_cats:
            report.append("\n  最准确:")
            for cat, acc in sorted_cats[:min(10, len(sorted_cats))]:
                report.append(f"    {cat}: {acc:.2%}")

            if len(sorted_cats) > 10:
                report.append("\n  最差:")
                for cat, acc in sorted_cats[-min(10, len(sorted_cats)):]:
                    report.append(f"    {cat}: {acc:.2%}")

        # 错误案例
        if self.errors:
            report.append(f"\n错误案例 (共 {len(self.errors)} 个):")

            # 按置信度排序，找出高置信度的错误
            high_conf_errors = sorted(
                [e for e in self.errors if e.get('confidence', 0) > 0.7],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )

            if high_conf_errors:
                report.append("\n  高置信度错误 (>0.7):")
                for err in high_conf_errors[:5]:
                    report.append(f"    - {err['product']}")
                    report.append(f"      预测: {err['predicted']} (置信度: {err.get('confidence', 0):.2f})")
                    report.append(f"      真实: {err['ground_truth']}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_errors(self, output_path: str):
        """保存错误案例到文件"""
        if self.errors:
            df = pd.DataFrame(self.errors)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"错误案例已保存到: {output_path}")


def load_test_data(csv_path: str) -> List[Dict]:
    """
    加载测试数据

    Args:
        csv_path: CSV文件路径，需要包含 '商品名称' 和分类列

    Returns:
        测试数据列表
    """
    df = pd.read_csv(csv_path)

    # 检查必需的列
    required_cols = ['商品名称']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件必须包含 '{col}' 列")

    test_data = []
    for _, row in df.iterrows():
        test_data.append({
            '商品名称': row['商品名称'],
            'level_1': row.get('一级分类', ''),
            'level_2': row.get('二级分类', ''),
            'level_3': row.get('三级分类', ''),
            'level_4': row.get('四级分类', ''),
        })

    return test_data


def run_evaluation(system, test_data: List[Dict], level: int = 4) -> Dict:
    """
    运行评估

    Args:
        system: 分类系统实例
        test_data: 测试数据
        level: 评估层级

    Returns:
        评估指标
    """
    evaluator = ClassificationEvaluator()

    logger.info(f"开始评估 {len(test_data)} 个样本...")

    predictions = []
    for i, item in enumerate(test_data, 1):
        product_name = item['商品名称']

        try:
            result = system.classify_single(product_name)

            predictions.append({
                '商品名称': product_name,
                'level_1': result.get('level_1', ''),
                'level_2': result.get('level_2', ''),
                'level_3': result.get('level_3', ''),
                'level_4': result.get('level_4', ''),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', '')
            })

            if i % 50 == 0:
                logger.info(f"已处理 {i}/{len(test_data)} 个样本")

        except Exception as e:
            logger.error(f"处理失败 ({i}/{len(test_data)}): {product_name} - {e}")
            predictions.append({
                '商品名称': product_name,
                'level_1': '', 'level_2': '', 'level_3': '', 'level_4': '',
                'confidence': 0,
                'reasoning': f'Error: {str(e)}'
            })

    # 计算指标
    metrics = evaluator.evaluate_predictions(predictions, test_data, level)

    return metrics, evaluator, predictions


def main():
    """测试评估模块"""
    print("=" * 60)
    print("分类系统评估测试")
    print("=" * 60)

    # 模拟测试数据
    test_data = [
        {'商品名称': '可口可乐330ml罐装', 'level_1': '食品', 'level_2': '饮料', 'level_3': '有汽饮品', 'level_4': '碳酸饮料'},
        {'商品名称': '农夫山泉NFC橙汁1L', 'level_1': '食品', 'level_2': '饮料', 'level_3': '果汁饮料', 'level_4': 'NFC果汁'},
        {'商品名称': '伊利安慕希酸奶', 'level_1': '食品', 'level_2': '乳制品', 'level_3': '酸奶', 'level_4': '常温酸奶'},
    ]

    # 模拟预测结果
    predictions = [
        {'商品名称': '可口可乐330ml罐装', 'level_1': '食品', 'level_2': '饮料', 'level_3': '有汽饮品', 'level_4': '碳酸饮料', 'confidence': 0.92},
        {'商品名称': '农夫山泉NFC橙汁1L', 'level_1': '食品', 'level_2': '饮料', 'level_3': '果汁饮料', 'level_4': '果汁', 'confidence': 0.85},
        {'商品名称': '伊利安慕希酸奶', 'level_1': '食品', 'level_2': '乳制品', 'level_3': '酸奶', 'level_4': '酸奶', 'confidence': 0.78},
    ]

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate_predictions(predictions, test_data, level=4)

    report = evaluator.generate_report(metrics)
    print(report)


if __name__ == '__main__':
    main()
