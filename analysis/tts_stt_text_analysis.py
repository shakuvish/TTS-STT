#!/usr/bin/env python3
"""
TTS/STT Text Quality Analysis
============================

Comprehensive analysis of text preservation through TTS/STT pipeline.
Focus on text quality metrics, patterns, and insights across languages, 
sentence lengths, and processing pipelines.

Usage: python analysis/tts_stt_text_analysis.py
Run from project root directory.

Requirements:
pip install polars altair pandas nltk jiwer python-Levenshtein sacrebleu evaluate
"""

import polars as pl
import altair as alt
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import unicodedata
import warnings
warnings.filterwarnings('ignore')

# Import metric libraries
import jiwer
import Levenshtein
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import sacrebleu
import evaluate

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Configure Altair
alt.data_transformers.enable('json')

# Initialize Hugging Face evaluate metrics
try:
    hf_bleu = evaluate.load("bleu")
    hf_wer = evaluate.load("wer")
    print("Loaded Hugging Face evaluate metrics")
except Exception as e:
    print(f"Warning: Could not load HF evaluate metrics: {e}")
    hf_bleu = None
    hf_wer = None

class TextQualityAnalyzer:
    """Comprehensive text quality analysis for TTS/STT pipeline evaluation."""
    
    def __init__(self, data_path: str = "data/output/tts_ready_10_pairs_processed_output.jsonl"):
        self.data_path = data_path
        self.df = None
        self.metrics_df = None
        self.output_dir = Path("analysis/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pl.DataFrame:
        """Load and preprocess the JSONL data."""
        print("Loading data...")
        self.df = pl.read_ndjson(self.data_path)

        # Filter to only successful pipeline runs
        if "pipeline_status" in self.df.columns:
            before = self.df.shape[0]
            self.df = self.df.filter(pl.col("pipeline_status") == "success")
            after = self.df.shape[0]
            print(f"Filtered to pipeline_status == 'success': {after}/{before} records retained")

        # Fix bin sorting by padding single digits with zeros
        def fix_bin_format(bin_str):
            if bin_str == '80+':
                return bin_str
            # Handle ranges like '0-4', '5-9', '10-14'
            if '-' in bin_str:
                start, end = bin_str.split('-')
                return f"{start.zfill(2)}-{end.zfill(2)}"
            return bin_str

        self.df = self.df.with_columns([
            pl.col('bin').map_elements(fix_bin_format, return_dtype=pl.Utf8).alias('bin')
        ])

        # Replace "gemini_flash" with "gtts" in tts_model_name due to original dataset creation oversight
        # During original dataset creation, forgot to change the model name from "gemini_flash" to "gtts"
        self.df = self.df.with_columns([
            pl.when(pl.col("tts_model_name") == "gemini_flash")
              .then(pl.lit("gtts"))
              .otherwise(pl.col("tts_model_name"))
              .alias("tts_model_name")
        ])

        print(f"Loaded {self.df.shape[0]} records")
        print(f"Columns: {self.df.columns}")
        print(f"Languages: {self.df['lang'].unique().sort()}")
        print(f"Bins: {self.df['bin'].unique().sort()}")
        print(f"TTS models: {self.df['tts_model_name'].unique().sort()}")
        print(f"STT models: {self.df['stt_model_name'].unique().sort()}")
        return self.df
    
    def normalize_text(self, text: str, lang: str) -> str:
        """Normalize text for fair comparison."""
        if pd.isna(text) or text is None:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', str(text))
        
        # Convert to lowercase to avoid case-sensitivity issues
        text = text.lower()
        
        # Language-specific normalization
        if lang == 'hi':
            # Remove extra spaces around Hindi punctuation
            text = re.sub(r'\s*।\s*', '।', text)
            # Normalize multiple spaces
            text = re.sub(r'\s+', ' ', text)
        elif lang == 'en':
            # Handle common English punctuation
            text = re.sub(r'\s*\.\s*$', '.', text)
            # Normalize multiple spaces
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_comprehensive_metrics(self, original: str, output: str, lang: str) -> Dict[str, float]:
        """Calculate all text quality metrics using established libraries."""
        # Normalize texts
        orig_norm = self.normalize_text(original, lang)
        out_norm = self.normalize_text(output, lang)
        
        if not orig_norm or not out_norm:
            return self._get_empty_metrics()
        
        metrics = {}
        
        # 1. Exact Match
        metrics['exact_match'] = float(orig_norm.lower() == out_norm.lower())
        
        # 2. Levenshtein-based metrics
        try:
            char_distance = Levenshtein.distance(orig_norm, out_norm)
            metrics['char_edit_distance'] = char_distance
            metrics['char_edit_distance_norm'] = char_distance / max(len(orig_norm), 1)
            metrics['levenshtein_ratio'] = Levenshtein.ratio(orig_norm, out_norm)
        except Exception as e:
            print(f"Levenshtein error: {e}")
            metrics.update({'char_edit_distance': 0, 'char_edit_distance_norm': 0, 'levenshtein_ratio': 1.0})
        
        # 3. Word Error Rate (WER) and Character Error Rate (CER)
        try:
            metrics['wer'] = jiwer.wer(orig_norm, out_norm)
            metrics['cer'] = jiwer.cer(orig_norm, out_norm)
        except Exception as e:
            print(f"jiwer error: {e}")
            metrics.update({'wer': 0.0, 'cer': 0.0})
        
        # 4. BLEU Score (NLTK implementation)
        try:
            # Tokenize for BLEU
            orig_tokens = word_tokenize(orig_norm, language='english' if lang == 'en' else 'english')
            out_tokens = word_tokenize(out_norm, language='english' if lang == 'en' else 'english')
            
            if orig_tokens and out_tokens:
                smoothie = SmoothingFunction().method4
                metrics['bleu_nltk'] = sentence_bleu([orig_tokens], out_tokens, smoothing_function=smoothie)
            else:
                metrics['bleu_nltk'] = 0.0
        except Exception as e:
            print(f"NLTK BLEU error: {e}")
            metrics['bleu_nltk'] = 0.0
        
        # 5. SacreBLEU (more standardized)
        try:
            sacre_bleu = sacrebleu.sentence_bleu(out_norm, [orig_norm])
            metrics['bleu_sacre'] = sacre_bleu.score / 100.0  # Convert to 0-1 scale
        except Exception as e:
            print(f"SacreBLEU error: {e}")
            metrics['bleu_sacre'] = 0.0
        
        # 6. Hugging Face evaluate metrics (if available)
        if hf_bleu and hf_wer:
            try:
                hf_bleu_result = hf_bleu.compute(predictions=[out_norm], references=[[orig_norm]])
                metrics['bleu_hf'] = hf_bleu_result['bleu']
                
                hf_wer_result = hf_wer.compute(predictions=[out_norm], references=[orig_norm])
                metrics['wer_hf'] = hf_wer_result
            except Exception as e:
                print(f"HF evaluate error: {e}")
                metrics.update({'bleu_hf': 0.0, 'wer_hf': 0.0})
        else:
            metrics.update({'bleu_hf': 0.0, 'wer_hf': 0.0})
        
        # 7. Additional useful metrics
        try:
            # Length ratio
            metrics['length_ratio'] = len(out_norm) / max(len(orig_norm), 1)
            
            # Word-level edit distance using NLTK
            orig_words = orig_norm.split()
            out_words = out_norm.split()
            word_edit_dist = nltk.edit_distance(orig_words, out_words)
            metrics['word_edit_distance'] = word_edit_dist
            metrics['word_edit_distance_norm'] = word_edit_dist / max(len(orig_words), 1)
            
        except Exception as e:
            print(f"Additional metrics error: {e}")
            metrics.update({
                'length_ratio': 1.0,
                'word_edit_distance': 0,
                'word_edit_distance_norm': 0.0
            })
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for edge cases."""
        return {
            'exact_match': 0.0,
            'char_edit_distance': 0,
            'char_edit_distance_norm': 1.0,
            'levenshtein_ratio': 0.0,
            'wer': 1.0,
            'cer': 1.0,
            'bleu_nltk': 0.0,
            'bleu_sacre': 0.0,
            'bleu_hf': 0.0,
            'wer_hf': 1.0,
            'length_ratio': 0.0,
            'word_edit_distance': 0,
            'word_edit_distance_norm': 1.0
        }
    
    def calculate_text_metrics(self) -> pl.DataFrame:
        """Calculate all text quality metrics for the dataset."""
        print("Calculating comprehensive text quality metrics...")
        
        # Convert to pandas for easier row-wise operations
        df_pd = self.df.to_pandas()
        
        metrics_list = []
        
        for idx, row in df_pd.iterrows():
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(df_pd)} rows")
            
            original = row['text']
            output = row['stt_text']
            lang = row['lang']
            
            # Calculate all metrics
            text_metrics = self.calculate_comprehensive_metrics(original, output, lang)
            
            # Create pipeline identifier
            pipeline = f"{row['tts_model_name']}_{row['stt_model_name']}"
            
            # Combine all data
            metrics = {
                'id': row['id'],
                'lang': lang,
                'bin': row['bin'],
                'pipeline': pipeline,
                'tts_model': row['tts_model_name'],
                'stt_model': row['stt_model_name'],
                'original_text': original,
                'output_text': output,
                'original_length': len(str(original)) if original else 0,
                'output_length': len(str(output)) if output else 0,
                **text_metrics
            }
            
            metrics_list.append(metrics)
        
        self.metrics_df = pl.DataFrame(metrics_list)
        print(f"Calculated metrics for {self.metrics_df.shape[0]} records")
        return self.metrics_df
    
    def analyze_by_pipeline(self) -> Dict:
        """Analyze performance by processing pipeline."""
        print("Analyzing by pipeline...")
        
        pipeline_stats = (
            self.metrics_df
            .group_by(['lang', 'pipeline'])
            .agg([
                pl.col('exact_match').mean().alias('exact_match_rate'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer'),
                pl.col('bleu_nltk').mean().alias('avg_bleu_nltk'),
                pl.col('bleu_sacre').mean().alias('avg_bleu_sacre'),
                pl.col('char_edit_distance_norm').mean().alias('avg_char_edit_norm'),
                pl.col('word_edit_distance_norm').mean().alias('avg_word_edit_norm'),
                pl.col('exact_match').count().alias('total_samples')
            ])
            .sort(['lang', 'avg_wer'], descending=[False, True])
        )

        return {
            'pipeline_stats': pipeline_stats,
            'chart': self.create_pipeline_performance_chart(pipeline_stats)
        }
    
    def analyze_by_bin(self) -> Dict:
        """Analyze performance by sentence length bin."""
        print("Analyzing by bin...")
        
        bin_stats = (
            self.metrics_df
            .group_by(['lang', 'bin'])
            .agg([
                pl.col('exact_match').mean().alias('exact_match_rate'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer'),
                pl.col('bleu_nltk').mean().alias('avg_bleu_nltk'),
                pl.col('bleu_sacre').mean().alias('avg_bleu_sacre'),
                pl.col('original_length').mean().alias('avg_original_length'),
                pl.col('exact_match').count().alias('total_samples')
            ])
            .sort(['lang', 'bin'])
        )

        return {
            'bin_stats': bin_stats,
            'chart': self.create_bin_performance_chart(bin_stats)
        }
    
    def analyze_by_bin_and_pipeline(self) -> Dict:
        """
        Analyze performance by sentence length bin, split by both language and pipeline.
        Returns both the stats DataFrame and a dict of charts for each (lang, pipeline).
        """
        print("Analyzing by bin and pipeline...")

        bin_stats = (
            self.metrics_df
            .group_by(['lang', 'pipeline', 'bin'])
            .agg([
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer'),
                pl.col('bleu_nltk').mean().alias('avg_bleu_nltk'),
                pl.col('bleu_sacre').mean().alias('avg_bleu_sacre'),
                pl.col('exact_match').count().alias('total_samples')
            ])
            .sort(['lang', 'pipeline', 'bin'])
        )

        charts = self.create_bin_charts_by_pipeline(bin_stats)
        return {
            'bin_stats': bin_stats,
            'charts': charts
        }

    def create_bin_charts_by_pipeline(self, bin_stats: pl.DataFrame) -> dict:
        """
        For each (lang, pipeline), create a line+point chart of BLEU (nltk, sacre), WER, and CER vs. bin.
        Returns a dict: {(lang, pipeline): chart}
        """
        import pandas as pd
        chart_data = bin_stats.to_pandas()

        # Melt to long form for Altair
        chart_data = chart_data.melt(
            id_vars=["lang", "pipeline", "bin"],
            value_vars=["avg_wer", "avg_cer", "avg_bleu_nltk", "avg_bleu_sacre"],
            var_name="metric",
            value_name="score"
        )
        chart_data["metric"] = chart_data["metric"].replace({
            "avg_wer": "WER",
            "avg_cer": "CER",
            "avg_bleu_nltk": "BLEU (NLTK)",
            "avg_bleu_sacre": "BLEU (SacreBLEU)"
        })

        charts = {}
        for (lang, pipeline), group in chart_data.groupby(["lang", "pipeline"]):
            base = alt.Chart(group).mark_line(point=True).encode(
                x=alt.X("bin:O", title="Sentence Length Bin"),
                y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["bin", "metric", "score"]
            ).properties(
                width=600,
                height=400,
                title=f"{lang.upper()} – {pipeline}"
            )
            charts[f"{lang}_{pipeline}"] = base
        return charts

    def analyze_by_language(self) -> Dict:
        """Analyze performance differences between languages."""
        print("Analyzing by language...")
        
        lang_stats = (
            self.metrics_df
            .group_by('lang')
            .agg([
                pl.col('exact_match').mean().alias('exact_match_rate'),
                pl.col('levenshtein_ratio').mean().alias('avg_levenshtein_ratio'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer'),
                pl.col('bleu_nltk').mean().alias('avg_bleu_nltk'),
                pl.col('bleu_sacre').mean().alias('avg_bleu_sacre'),
                pl.col('exact_match').count().alias('total_samples')
            ])
        )
        
        return {
            'lang_stats': lang_stats,
            'chart': self.create_language_comparison_chart(lang_stats)
        }
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze common error patterns."""
        print("Analyzing error patterns...")
        
        # Get failed cases (non-exact matches)
        failed_cases = self.metrics_df.filter(pl.col('exact_match') == 0.0)
        
        # Sample error examples with various metrics
        error_examples = (
            failed_cases
            .sample(n=min(30, failed_cases.shape[0]))
            .select([
                'lang', 'pipeline', 'original_text', 'output_text', 
                'levenshtein_ratio', 'wer', 'cer', 'bleu_nltk'
            ])
            .sort('levenshtein_ratio')
        )
        
        # Error distribution by pipeline
        error_by_pipeline = (
            self.metrics_df
            .group_by(['lang', 'pipeline'])
            .agg([
                (pl.col('exact_match') == 0.0).sum().alias('error_count'),
                pl.col('exact_match').count().alias('total_count'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer')
            ])
            .with_columns([
                (pl.col('error_count') / pl.col('total_count')).alias('error_rate')
            ])
            .sort(['lang', 'error_rate'])
        )
        
        return {
            'error_examples': error_examples,
            'error_by_pipeline': error_by_pipeline,
            'chart': self.create_error_analysis_chart(error_by_pipeline)
        }
    
    def create_pipeline_performance_chart(self, pipeline_stats: pl.DataFrame) -> alt.Chart:
        """Create pipeline performance comparison chart (WER, CER, BLEU)."""
        chart_data = pipeline_stats.to_pandas()

        # WER chart
        wer_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('pipeline:N', title='Pipeline', sort='y'),
            y=alt.Y('avg_wer:Q', title='Average WER (lower is better)'),
            color=alt.Color('lang:N', title='Language'),
            column=alt.Column('lang:N', title='Language'),
            tooltip=['pipeline', 'avg_wer', 'avg_cer', 'avg_bleu_nltk', 'avg_bleu_sacre', 'exact_match_rate']
        ).properties(
            width=1200,
            height=800,
            title='Pipeline Performance: Word Error Rate'
        )

        # CER chart
        cer_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('pipeline:N', title='Pipeline', sort='y'),
            y=alt.Y('avg_cer:Q', title='Average CER (lower is better)'),
            color=alt.Color('lang:N', title='Language'),
            column=alt.Column('lang:N', title='Language'),
            tooltip=['pipeline', 'avg_cer', 'avg_wer', 'avg_bleu_nltk', 'avg_bleu_sacre', 'exact_match_rate']
        ).properties(
            width=1200,
            height=800,
            title='Pipeline Performance: Character Error Rate'
        )

        # BLEU chart (SacreBLEU)
        bleu_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('pipeline:N', title='Pipeline', sort='-y'),
            y=alt.Y('avg_bleu_sacre:Q', title='Average BLEU (SacreBLEU, higher is better)', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('lang:N', title='Language'),
            column=alt.Column('lang:N', title='Language'),
            tooltip=['pipeline', 'avg_bleu_sacre', 'avg_bleu_nltk', 'avg_wer', 'avg_cer', 'exact_match_rate']
        ).properties(
            width=1200,
            height=800,
            title='Pipeline Performance: BLEU (SacreBLEU)'
        )

        return wer_chart & cer_chart & bleu_chart
    

    def create_bin_performance_chart(self, bin_stats: pl.DataFrame) -> alt.Chart:
        """
        Create a performance chart vs sentence length, faceted by language,
        with a legend for BLEU (nltk, sacre), WER, and CER scores.
        """

        # --- Step 1: Convert Polars DataFrame to pandas and reshape ---
        chart_data = bin_stats.to_pandas()

        # Melt (unpivot) from wide to long format for Altair
        chart_data = chart_data.melt(
            id_vars=["bin", "lang"],                      # keep bin and lang as-is
            value_vars=["avg_wer", "avg_cer", "avg_bleu_nltk", "avg_bleu_sacre"],  # convert these to rows
            var_name="metric",                            # new column for metric name
            value_name="score"                            # new column for value
        )

        # --- Step 2: Rename metric values for prettier legend labels ---
        chart_data["metric"] = chart_data["metric"].replace({
            "avg_wer": "WER",
            "avg_cer": "CER",
            "avg_bleu_nltk": "BLEU (NLTK)",
            "avg_bleu_sacre": "BLEU (SacreBLEU)"
        })

        # --- Step 3: Create base chart ---
        base = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("bin:O", title="Sentence Length Bin"),         # x-axis: sentence length bins
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),  # y-axis
            color=alt.Color("metric:N", title="Metric")            # color by metric with legend
        ).properties(
            width=800,  # width of each facet
            height=400  # height of each facet
        )

        # --- Step 4: Facet by language ---
        chart = base.facet(
            column=alt.Column("lang:N", title="Language")          # facet by lang: "en", "hi"
        ).resolve_scale(
            y="independent"                                        # allow separate y-scales
        ).properties(
            title="Performance vs Sentence Length"
        )

        return chart

    def create_language_comparison_chart(self, lang_stats: pl.DataFrame) -> alt.Chart:
        """Create comprehensive language comparison chart."""
        chart_data = lang_stats.to_pandas()
        
        # Reshape data for multi-metric comparison
        metrics_to_plot = ['exact_match_rate', 'avg_levenshtein_ratio', 'avg_bleu_nltk']
        plot_data = []
        
        for _, row in chart_data.iterrows():
            for metric in metrics_to_plot:
                plot_data.append({
                    'lang': row['lang'],
                    'metric': metric,
                    'value': row[metric]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('metric:N', title='Metric'),
            y=alt.Y('value:Q', title='Score'),
            color=alt.Color('lang:N', title='Language'),
            column=alt.Column('lang:N', title='Language'),
            tooltip=['lang', 'metric', 'value']
        ).properties(
            width=800,
            height=800,
            title='Language Comparison: Multiple Metrics'
        )
        
        return chart
    
    def create_error_analysis_chart(self, error_by_pipeline: pl.DataFrame) -> alt.Chart:
        """Create comprehensive error analysis chart."""
        chart_data = error_by_pipeline.to_pandas()
        
        # Error rate chart
        error_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('pipeline:N', title='Pipeline'),
            y=alt.Y('error_rate:Q', title='Error Rate'),
            color=alt.Color('lang:N', title='Language'),
            column=alt.Column('lang:N', title='Language'),
            tooltip=['pipeline', 'error_rate', 'avg_wer', 'avg_cer']
        ).properties(
            width=1200,
            height=800,
            title='Error Rate by Pipeline'
        )
        
        # WER/CER scatter
        scatter_chart = alt.Chart(chart_data).mark_circle(size=100).encode(
            x=alt.X('avg_wer:Q', title='Average WER'),
            y=alt.Y('avg_cer:Q', title='Average CER'),
            color=alt.Color('lang:N', title='Language'),
            tooltip=['pipeline', 'avg_wer', 'avg_cer', 'error_rate']
        ).properties(
            width=1600,
            height=1200,
            title='WER vs CER by Pipeline'
        )
        
        return error_chart & scatter_chart
    
    def create_comprehensive_heatmap(self) -> alt.Chart:
        """Create comprehensive performance heatmap."""
        # Prepare data for heatmap
        heatmap_data = (
            self.metrics_df
            .group_by(['lang', 'pipeline', 'bin'])
            .agg([
                pl.col('levenshtein_ratio').mean().alias('avg_levenshtein_ratio'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('exact_match').mean().alias('exact_match_rate')
            ])
        ).to_pandas()
        
        # Levenshtein ratio heatmap
        lev_heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('bin:O', title='Sentence Length Bin'),
            y=alt.Y('pipeline:N', title='Pipeline'),
            color=alt.Color('avg_levenshtein_ratio:Q', 
                          title='Avg Levenshtein Ratio',
                          scale=alt.Scale(scheme='viridis')),
            facet=alt.Facet('lang:N', title='Language'),
            tooltip=['lang', 'pipeline', 'bin', 'avg_levenshtein_ratio', 'avg_wer']
        ).properties(
            width=800,
            height=800,
            title='Performance Heatmap: Levenshtein Ratio'
        )
        
        # WER heatmap (inverted colors since lower is better)
        wer_heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('bin:O', title='Sentence Length Bin'),
            y=alt.Y('pipeline:N', title='Pipeline'),
            color=alt.Color('avg_wer:Q', 
                          title='Avg WER',
                          scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
            facet=alt.Facet('lang:N', title='Language'),
            tooltip=['lang', 'pipeline', 'bin', 'avg_wer', 'avg_levenshtein_ratio']
        ).properties(
            width=800,
            height=800,
            title='Performance Heatmap: Word Error Rate'
        )
        
        return lev_heatmap & wer_heatmap
    
    def generate_summary_tables(self) -> Dict[str, pl.DataFrame]:
        """Generate comprehensive summary tables for research report."""
        print("Generating summary tables...")
        
        # Overall performance summary with all metrics
        overall_summary = (
            self.metrics_df
            .group_by(['lang', 'pipeline'])
            .agg([
                pl.col('exact_match').mean().alias('exact_match_rate'),
                pl.col('levenshtein_ratio').mean().alias('levenshtein_ratio'),
                pl.col('wer').mean().alias('word_error_rate'),
                pl.col('cer').mean().alias('char_error_rate'),
                pl.col('bleu_nltk').mean().alias('bleu_nltk'),
                pl.col('bleu_sacre').mean().alias('bleu_sacre'),
                pl.col('char_edit_distance_norm').mean().alias('char_edit_norm'),
                pl.col('word_edit_distance_norm').mean().alias('word_edit_norm'),
                pl.col('length_ratio').mean().alias('avg_length_ratio'),
                pl.col('exact_match').count().alias('samples')
            ])
            .sort(['lang', 'levenshtein_ratio'], descending=[False, True])
        )
        
        # Best performing combinations by different metrics
        best_exact_match = (
            overall_summary
            .group_by('lang')
            .agg([
                pl.col('exact_match_rate').max().alias('best_exact_match_rate'),
                pl.col('pipeline').first().alias('best_exact_match_pipeline')
            ])
        )
        
        best_levenshtein = (
            overall_summary
            .group_by('lang')
            .agg([
                pl.col('levenshtein_ratio').max().alias('best_levenshtein_ratio'),
                pl.col('pipeline').first().alias('best_levenshtein_pipeline')
            ])
        )
        
        best_wer = (
            overall_summary
            .group_by('lang')
            .agg([
                pl.col('word_error_rate').min().alias('best_wer'),
                pl.col('pipeline').first().alias('best_wer_pipeline')
            ])
        )
        
        # Performance by bin summary
        bin_summary = (
            self.metrics_df
            .group_by(['lang', 'bin'])
            .agg([
                pl.col('levenshtein_ratio').mean().alias('avg_levenshtein_ratio'),
                pl.col('exact_match').mean().alias('exact_match_rate'),
                pl.col('wer').mean().alias('avg_wer'),
                pl.col('cer').mean().alias('avg_cer'),
                pl.col('bleu_nltk').mean().alias('avg_bleu'),
                pl.col('original_length').mean().alias('avg_length')
            ])
            .sort(['lang', 'bin'])
        )
        
        # Metric correlations
        correlation_data = self.metrics_df.select([
            'exact_match', 'levenshtein_ratio', 'wer', 'cer', 
            'bleu_nltk', 'bleu_sacre', 'char_edit_distance_norm'
        ]).to_pandas().corr()
        correlation_df = pl.DataFrame(correlation_data.reset_index())
        
        return {
            'overall_summary': overall_summary,
            'best_exact_match': best_exact_match,
            'best_levenshtein': best_levenshtein,
            'best_wer': best_wer,
            'bin_summary': bin_summary,
            'metric_correlations': correlation_df
        }
    
    def save_results(self, results: Dict):
        """Save all results to files."""
        print("Saving results...")

        import shutil

        # Prepare output directories
        plots_dir = self.output_dir / "plots"
        html_dir = self.output_dir / "html_files"
        csv_dir = self.output_dir / "csv_files"
        plots_dir.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Save summary tables
        for name, df in results['tables'].items():
            df.write_csv(self.output_dir / f"{name}.csv")

        # Save detailed metrics
        self.metrics_df.write_csv(self.output_dir / "detailed_metrics.csv")

        # Save main charts as HTML and PNG
        charts = {
            'pipeline_performance': results['pipeline_analysis']['chart'],
            'bin_performance': results['bin_analysis']['chart'],
            'language_comparison': results['language_analysis']['chart'],
            'error_analysis': results['error_analysis']['chart'],
            'comprehensive_heatmap': results['heatmap']
        }

        for name, chart in charts.items():
            html_path = self.output_dir / f"{name}.html"
            png_path = plots_dir / f"{name}.png"
            chart.save(str(html_path))
            try:
                chart.save(str(png_path), format="png")
            except Exception as e:
                print(f"Could not save {name} as PNG: {e}")

        # Save per-(lang, pipeline) bin charts
        bin_pipeline_charts = results.get('bin_pipeline_analysis', {}).get('charts', {})
        for key, chart in bin_pipeline_charts.items():
            html_path = self.output_dir / f"bin_performance_{key}.html"
            png_path = plots_dir / f"bin_performance_{key}.png"
            chart.save(str(html_path))
            try:
                chart.save(str(png_path), format="png")
            except Exception as e:
                print(f"Could not save bin_performance_{key} as PNG: {e}")

        # Move HTML files to html_files, CSV files to csv_files
        for file in self.output_dir.glob("*.html"):
            shutil.move(str(file), str(html_dir / file.name))
        for file in self.output_dir.glob("*.csv"):
            shutil.move(str(file), str(csv_dir / file.name))

        print(f"Results saved to {self.output_dir}")
        print(f"HTML files moved to {html_dir}")
        print(f"CSV files moved to {csv_dir}")
        print(f"PNG plots saved to {plots_dir}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting TTS/STT Text Quality Analysis with Professional Metrics")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Calculate metrics
        self.calculate_text_metrics()
        
        # Run analyses
        pipeline_analysis = self.analyze_by_pipeline()
        bin_analysis = self.analyze_by_bin()
        bin_pipeline_analysis = self.analyze_by_bin_and_pipeline()
        language_analysis = self.analyze_by_language()
        error_analysis = self.analyze_error_patterns()
        
        # Create comprehensive visualizations
        heatmap = self.create_comprehensive_heatmap()
        
        # Generate tables
        summary_tables = self.generate_summary_tables()
        
        # Compile results
        results = {
            'pipeline_analysis': pipeline_analysis,
            'bin_analysis': bin_analysis,
            'bin_pipeline_analysis': bin_pipeline_analysis,
            'language_analysis': language_analysis,
            'error_analysis': error_analysis,
            'heatmap': heatmap,
            'tables': summary_tables
        }
        
        # Save results
        self.save_results(results)
        
        # Print key insights
        self.print_key_insights(results)
        
        return results
    
    def print_key_insights(self, results: Dict):
        """Print key insights from the analysis."""
        print("\n" + "=" * 70)
        print("KEY INSIGHTS")
        print("=" * 70)
        
        # Best performing pipelines by different metrics
        best_lev = results['tables']['best_levenshtein']
        best_wer = results['tables']['best_wer']
        best_exact = results['tables']['best_exact_match']
        
        print(f"\nBEST PERFORMING PIPELINES:")
        print("By Levenshtein Ratio (similarity):")
        for row in best_lev.iter_rows(named=True):
            print(f"  {row['lang'].upper()}: {row['best_levenshtein_pipeline']} "
                  f"(Score: {row['best_levenshtein_ratio']:.3f})")
        
        print("\nBy Word Error Rate (lower is better):")
        for row in best_wer.iter_rows(named=True):
            print(f"  {row['lang'].upper()}: {row['best_wer_pipeline']} "
                  f"(WER: {row['best_wer']:.3f})")
        
        print("\nBy Exact Match Rate:")
        for row in best_exact.iter_rows(named=True):
            print(f"  {row['lang'].upper()}: {row['best_exact_match_pipeline']} "
                  f"(Rate: {row['best_exact_match_rate']:.3f})")
        
        # Language comparison
        lang_stats = results['language_analysis']['lang_stats']
        print(f"\nLANGUAGE COMPARISON:")
        for row in lang_stats.iter_rows(named=True):
            print(f"{row['lang'].upper()}: "
                  f"Similarity: {row['avg_levenshtein_ratio']:.3f}, "
                  f"WER: {row['avg_wer']:.3f}, "
                  f"CER: {row['avg_cer']:.3f}, "
                  f"Exact Match: {row['exact_match_rate']:.3f}")
        
        # Sample error cases
        error_examples = results['error_analysis']['error_examples']
        print(f"\nSAMPLE ERROR CASES (Worst performing):")
        for i, row in enumerate(error_examples.head(3).iter_rows(named=True)):
            print(f"\n{i+1}. {row['lang'].upper()} - {row['pipeline']}")
            print(f"   Original: {row['original_text']}")
            print(f"   Output:   {row['output_text']}")
            print(f"   Similarity: {row['levenshtein_ratio']:.3f}, "
                  f"WER: {row['wer']:.3f}, "
                  f"CER: {row['cer']:.3f}")
        
        print(f"\nANALYSIS COMPLETE!")
        print(f"📊 Results saved to: {self.output_dir}")
        print(f"📈 Charts: HTML files for interactive visualization")
        print(f"📋 Tables: CSV files for research report")


def main():
    """Main function to run the analysis."""
    analyzer = TextQualityAnalyzer()
    results = analyzer.run_full_analysis()
    return results


if __name__ == "__main__":
    main()
