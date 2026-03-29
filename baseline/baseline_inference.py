"""
Baseline inference for PR Code Review Assistant.

This script runs baseline evaluation using various LLM providers.
Supports: OpenAI GPT-4, Groq (GPT OSS 120B), and more.
"""

import os
import json
import re
import requests
import argparse
from typing import Dict, List, Any, Optional
from openai import OpenAI

# System prompt for code review
SYSTEM_PROMPT = """You are an expert code reviewer specializing in security, code quality, and best practices.

Your task is to review pull request changes and identify:
1. Security vulnerabilities (SQL injection, XSS, authentication issues, etc.)
2. Logic bugs and edge cases
3. Code quality issues (style, maintainability, performance)

For each issue you find, provide:
- Exact file path and line number
- Clear description of the problem
- Severity level (info, warning, error, or critical)
- Category (security, bug, performance, style, maintainability, testing, or documentation)
- Suggested fix if possible

Your response must be valid JSON matching this schema:
{
  "inline_comments": [
    {
      "file_path": "string",
      "line_number": integer,
      "comment": "string",
      "severity": "info|warning|error|critical",
      "category": "security|bug|performance|style|maintainability|testing|documentation",
      "suggested_fix": "string (optional)"
    }
  ],
  "general_comments": [
    {
      "comment": "string",
      "category": "architecture|approach|testing|documentation|general"
    }
  ],
  "decision": {
    "decision": "approve|request_changes|comment",
    "summary": "string"
  }
}

Be thorough but precise. Focus on real issues, not nitpicks.

IMPORTANT OUTPUT LIMITS:
- Return at most 8 inline comments total (highest impact only).
- Keep each comment under 220 characters.
- Return strictly valid JSON with double quotes and no trailing text."""


class BaselineAgent:
    """Baseline agent supporting multiple LLM providers."""

    # Provider configurations
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4-turbo-preview"
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "env_key": "GROQ_API_KEY",
            "default_model": "openai/gpt-oss-120b"
        }
    }
    VALID_SEVERITIES = {"info", "warning", "error", "critical"}
    VALID_INLINE_CATEGORIES = {"security", "bug", "performance", "style", "maintainability", "testing", "documentation"}
    VALID_GENERAL_CATEGORIES = {"architecture", "approach", "testing", "documentation", "general"}
    VALID_DECISIONS = {"approve", "request_changes", "comment"}

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        env_base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        """
        Initialize baseline agent.

        Args:
            provider: LLM provider ("openai" or "groq")
            model: Model name (uses provider default if None)
            env_base_url: Base URL for the environment server
            api_key: API key (uses env variable if None)
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(self.PROVIDERS.keys())}")

        provider_config = self.PROVIDERS[provider]

        # Get API key
        self.api_key = api_key or os.getenv(provider_config["env_key"])
        if not self.api_key:
            raise ValueError(
                f"{provider_config['env_key']} environment variable not set. "
                f"Please set it with: export {provider_config['env_key']}='your-key-here'"
            )

        # Initialize OpenAI client with provider's base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=provider_config["base_url"]
        )

        self.provider = provider
        self.model = model or provider_config["default_model"]
        self.env_base_url = env_base_url
        self.http = requests.Session()

        print(f"✓ Initialized {provider.upper()} baseline")
        print(f"  Model: {self.model}")
        print(f"  Environment: {env_base_url}")
        print()

    def _build_prompt(self, pr_state: Dict[str, Any]) -> str:
        """Build prompt from PR state."""
        prompt = f"""Review this pull request:

PR #{pr_state['pr_id']}: {pr_state['metadata']['title']}
Description: {pr_state['metadata']['description']}
Author: {pr_state['metadata']['author']}

Changes:
"""

        for file in pr_state['files']:
            prompt += f"\n\n## File: {file['path']} ({file['language']})\n"
            prompt += f"Additions: {len(file['additions'])}, Deletions: {len(file['deletions'])}\n\n"

            if file['context']:
                prompt += "Context:\n"
                for line in file['context']:
                    prompt += f"  {line}\n"

            if file['additions']:
                prompt += "\nAdded lines:\n"
                for line in file['additions']:
                    prompt += f"+ {line}\n"

            if file['deletions']:
                prompt += "\nDeleted lines:\n"
                for line in file['deletions']:
                    prompt += f"- {line}\n"

        return prompt

    def review_pr(self, pr_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate review using LLM.

        Args:
            pr_state: PR state from environment

        Returns:
            Action dictionary
        """
        prompt = self._build_prompt(pr_state)

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 4096
            }

            # OpenAI-compatible providers support json_object mode.
            if self.provider in {"openai", "groq"}:
                api_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**api_params)

            # Parse response
            content = response.choices[0].message.content

            # Try to extract JSON from response
            action = self._parse_json_response(content)
            return action

        except Exception as e:
            print(f"Error calling {self.provider.upper()}: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal valid action
            return {
                "inline_comments": [],
                "general_comments": [],
                "decision": {
                    "decision": "comment",
                    "summary": f"Error during review: {e}"
                }
            }

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            # Try direct parsing first
            return json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    def _extract_observation(self, payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """
        Extract observation object from API payload.

        OpenEnv servers may return either:
        1) raw observation: {"pr_state": ..., "feedback": ..., "metadata": ...}
        2) wrapped response: {"observation": {"pr_state": ...}, ...}
        """
        if isinstance(payload, dict):
            if "pr_state" in payload:
                return payload
            if isinstance(payload.get("observation"), dict):
                return payload["observation"]

        raise ValueError(
            f"Unexpected {endpoint} response format. "
            f"Top-level keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    def _parse_line_number(self, value: Any) -> Optional[int]:
        """Convert LLM-provided line number to int if possible."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM output into schema-valid Action payload."""
        if not isinstance(action, dict):
            action = {}

        normalized_inline = []
        for comment in action.get("inline_comments", []) or []:
            if not isinstance(comment, dict):
                continue

            file_path = comment.get("file_path")
            text = comment.get("comment")
            line_number = self._parse_line_number(comment.get("line_number"))
            severity = str(comment.get("severity", "warning")).lower()
            category = str(comment.get("category", "bug")).lower()

            if not isinstance(file_path, str) or not file_path.strip():
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            if line_number is None:
                continue
            if severity not in self.VALID_SEVERITIES:
                severity = "warning"
            if category not in self.VALID_INLINE_CATEGORIES:
                category = "bug"

            item = {
                "file_path": file_path.strip(),
                "line_number": line_number,
                "comment": text.strip(),
                "severity": severity,
                "category": category,
            }
            suggested_fix = comment.get("suggested_fix")
            if isinstance(suggested_fix, str) and suggested_fix.strip():
                item["suggested_fix"] = suggested_fix.strip()
            normalized_inline.append(item)

        normalized_general = []
        for comment in action.get("general_comments", []) or []:
            if not isinstance(comment, dict):
                continue
            text = comment.get("comment")
            category = str(comment.get("category", "general")).lower()
            if not isinstance(text, str) or not text.strip():
                continue
            if category not in self.VALID_GENERAL_CATEGORIES:
                category = "general"
            normalized_general.append({
                "comment": text.strip(),
                "category": category,
            })

        decision_obj = action.get("decision", {})
        if not isinstance(decision_obj, dict):
            decision_obj = {}
        decision = str(decision_obj.get("decision", "comment")).lower()
        if decision not in self.VALID_DECISIONS:
            decision = "comment"
        summary = decision_obj.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "Automated review generated by baseline agent."

        return {
            "inline_comments": normalized_inline,
            "general_comments": normalized_general,
            "decision": {
                "decision": decision,
                "summary": summary.strip(),
            },
        }

    def run_evaluation(self, task_ids: List[str]) -> Dict[str, Any]:
        """
        Run baseline on all tasks.

        Args:
            task_ids: List of task IDs to evaluate

        Returns:
            Results dictionary with scores
        """
        results = {}

        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_id}")
            print(f"{'='*60}")

            try:
                # Reset environment to this task
                reset_response = self.http.post(
                    f"{self.env_base_url}/reset",
                    json={"task_id": task_id}
                )
                reset_response.raise_for_status()
                reset_payload = reset_response.json()
                obs = self._extract_observation(reset_payload, "/reset")

                # Generate review
                print(f"Generating review with {self.provider.upper()} ({self.model})...")
                action = self._normalize_action(self.review_pr(obs["pr_state"]))

                print(f"Found {len(action['inline_comments'])} inline comments")
                print(f"Decision: {action['decision']['decision']}")

                # Execute step
                step_response = self.http.post(
                    f"{self.env_base_url}/step",
                    json={"action": action}
                )
                step_response.raise_for_status()
                step_payload = step_response.json()
                result = self._extract_observation(step_payload, "/step")

                # Extract results
                feedback = result.get("feedback", {})
                passed = result.get("metadata", {}).get("passed", False)

                results[task_id] = {
                    "score": feedback.get("score", 0.0),
                    "passed": passed,
                    "precision": feedback.get("precision", 0.0),
                    "recall": feedback.get("recall", 0.0),
                    "severity_alignment": feedback.get("severity_alignment", 0.0),
                    "true_positives": feedback.get("true_positives", 0),
                    "false_positives": feedback.get("false_positives", 0),
                    "false_negatives": feedback.get("false_negatives", 0)
                }

                print(f"\nResults:")
                print(f"  Score: {results[task_id]['score']:.2f}")
                print(f"  Passed: {'✓' if passed else '✗'}")
                print(f"  Precision: {results[task_id]['precision']:.2f}")
                print(f"  Recall: {results[task_id]['recall']:.2f}")

            except Exception as e:
                print(f"Error evaluating {task_id}: {e}")
                results[task_id] = {
                    "error": str(e),
                    "score": 0.0,
                    "passed": False
                }

        # Calculate average
        valid_scores = [r["score"] for r in results.values() if "error" not in r]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        print(f"\n{'='*60}")
        print(f"AVERAGE SCORE: {avg_score:.2f}")
        print(f"{'='*60}\n")

        return {
            "results": results,
            "average_score": avg_score,
            "model": self.model
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline inference for PR code review")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "groq"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (overrides provider default)"
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="http://localhost:8000",
        help="Environment server base URL"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=["task1_security_basic", "task2_quality_logic", "task3_advanced_review"],
        help="Task IDs to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Initialize baseline agent
    baseline = BaselineAgent(
        provider=args.provider,
        model=args.model,
        env_base_url=args.env_url
    )

    # Run evaluation
    results = baseline.run_evaluation(args.task_ids)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
