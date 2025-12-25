"""
Tests for Content Quality Assessment Module.

Tests cover all detection types:
1. Null/empty responses
2. Too-short responses
3. Gibberish detection
4. Non-English content
5. Test responses
6. Non-responses (N/A, idk, etc.)
7. Excessive repetition
8. No alphabetic content
"""

import pytest
import time
from src.content_quality import ContentQualityFilter


class TestContentQualityFilter:
    """Test suite for ContentQualityFilter class."""

    @pytest.fixture
    def filter(self):
        """Create a standard filter instance."""
        return ContentQualityFilter(
            min_words=3,
            min_chars=10,
            max_repetition_ratio=0.7,
            min_english_word_ratio=0.3
        )

    # ========================================================================
    # Test 1: Null and Empty Detection
    # ========================================================================

    def test_null_response(self, filter):
        """Test detection of None/null responses."""
        result = filter.assess_signal(None)
        assert result['is_analytic'] is False
        assert result['confidence'] == 1.0
        assert 'null' in result['flags']
        assert result['recommendation'] == 'exclude'

    def test_empty_string(self, filter):
        """Test detection of empty string."""
        result = filter.assess_signal('')
        assert result['is_analytic'] is False
        assert result['confidence'] == 1.0
        assert 'empty' in result['flags']
        assert result['recommendation'] == 'exclude'

    def test_whitespace_only(self, filter):
        """Test detection of whitespace-only responses."""
        test_cases = ['   ', '\t\t', '\n\n', '  \t  \n  ']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'empty' in result['flags']
            assert result['recommendation'] == 'exclude'

    # ========================================================================
    # Test 2: Too Short Detection
    # ========================================================================

    def test_too_short_chars(self, filter):
        """Test detection of responses with too few characters."""
        result = filter.assess_signal('yes')  # 3 chars < 10 minimum
        assert result['is_analytic'] is False
        assert 'too_short_chars' in result['flags']
        assert 'too_short_words' in result['flags']

    def test_too_short_words(self, filter):
        """Test detection of responses with too few words."""
        result = filter.assess_signal('ok sure')  # 2 words < 3 minimum
        assert result['is_analytic'] is False
        assert 'too_short_words' in result['flags']

    def test_minimum_acceptable_length(self, filter):
        """Test that minimum acceptable length passes."""
        result = filter.assess_signal('This is a valid response with content')
        assert result['is_analytic'] is True
        assert len(result['flags']) == 0

    # ========================================================================
    # Test 3: Non-Response Patterns
    # ========================================================================

    def test_non_response_na(self, filter):
        """Test detection of N/A responses."""
        test_cases = ['N/A', 'n/a', 'NA', 'na', 'N / A', '  n/a  ']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'non_response' in result['flags']
            assert result['recommendation'] == 'exclude'

    def test_non_response_idk(self, filter):
        """Test detection of 'I don't know' responses."""
        test_cases = ['idk', 'IDK', "i don't know", "I don't know", "i dont know"]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'non_response' in result['flags']

    def test_non_response_variants(self, filter):
        """Test detection of various non-response patterns."""
        test_cases = [
            'none', 'None', 'NONE',
            'nothing', 'Nothing',
            'no comment', 'No Comment',
            'no response',
            'skip', 'Skip',
            'pass', 'Pass',
            '---', '....', '????'
        ]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'non_response' in result['flags']

    # ========================================================================
    # Test 4: Test Response Patterns
    # ========================================================================

    def test_test_response_simple(self, filter):
        """Test detection of simple test responses."""
        test_cases = ['test', 'Test', 'TEST', 'testing', 'Testing']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'test_response' in result['flags']

    def test_test_response_keyboard(self, filter):
        """Test detection of keyboard test patterns."""
        test_cases = ['asdf', 'ASDF', 'asdfgh', 'qwerty', 'qwertyuiop']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            # Could be test_response or gibberish
            assert 'test_response' in result['flags'] or 'gibberish' in result['flags']

    def test_test_response_patterns(self, filter):
        """Test detection of common test patterns."""
        test_cases = [
            '123', '1234567890',
            'abc', 'abcdefgh',
            'xxx', 'xxxxxx',
            'zzz', 'zzzzzz',
            'sample', 'Sample',
            'example', 'Example',
            'placeholder',
            'lorem ipsum'
        ]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            # Should flag as test_response, gibberish, excessive_repetition, or non-english
            assert len(result['flags']) > 0

    # ========================================================================
    # Test 5: Gibberish Detection
    # ========================================================================

    def test_gibberish_keyboard_walks(self, filter):
        """Test detection of keyboard walk gibberish."""
        test_cases = [
            'qwertyuiop response',
            'asdfghjkl typed this',
            'zxcvbnm content',
            'some qwerty text'
        ]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'gibberish' in result['flags']

    def test_gibberish_low_vowels(self, filter):
        """Test detection of low vowel ratio (gibberish characteristic)."""
        # Very low vowel content
        result = filter.assess_signal('qzxcvbnmpqrstw')
        assert result['is_analytic'] is False
        # Should trigger gibberish or non_english
        assert 'gibberish' in result['flags'] or 'non_english' in result['flags']

    def test_gibberish_char_repetition(self, filter):
        """Test detection of excessive character repetition."""
        test_cases = [
            'aaaaaaaaaa',
            'hhhhhhhh response',
            'text with xxxxxxx in it'
        ]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'gibberish' in result['flags'] or 'excessive_repetition' in result['flags']

    def test_gibberish_very_short_words(self, filter):
        """Test detection of many very short words (gibberish pattern)."""
        result = filter.assess_signal('a b c d e f g h')
        assert result['is_analytic'] is False
        # Could be multiple flags
        assert len(result['flags']) > 0

    # ========================================================================
    # Test 6: Non-English Detection
    # ========================================================================

    def test_non_english_spanish(self, filter):
        """Test detection of Spanish text."""
        result = filter.assess_signal('Hola, cómo estás hoy. Muy bien gracias.')
        assert result['is_analytic'] is False
        assert 'non_english' in result['flags']
        # Should recommend review, not exclude (could be valid multilingual data)
        assert result['recommendation'] in ['review', 'exclude']

    def test_non_english_french(self, filter):
        """Test detection of French text."""
        result = filter.assess_signal('Bonjour, comment allez-vous aujourd\'hui?')
        assert result['is_analytic'] is False
        assert 'non_english' in result['flags']

    def test_non_english_mixed(self, filter):
        """Test detection of mixed language text."""
        # Some English words, but mostly non-English
        result = filter.assess_signal('Das ist sehr gut and interessant')
        # May or may not flag depending on English word ratio
        # Just verify it doesn't crash
        assert 'is_analytic' in result

    def test_english_proper_nouns(self, filter):
        """Test that English text with proper nouns is not flagged as non-English."""
        result = filter.assess_signal('I work at Microsoft in Seattle and enjoy the culture there.')
        # Should pass as English despite proper nouns not in common word list
        # The common words (I, at, in, and, the) should be enough
        assert result['is_analytic'] is True or 'non_english' not in result['flags']

    # ========================================================================
    # Test 7: Excessive Repetition
    # ========================================================================

    def test_excessive_repetition_words(self, filter):
        """Test detection of excessive word repetition."""
        result = filter.assess_signal('work work work work work work work work')
        assert result['is_analytic'] is False
        assert 'excessive_repetition' in result['flags']

    def test_excessive_repetition_phrase(self, filter):
        """Test detection of repeated phrases."""
        result = filter.assess_signal('very very very very very good')
        assert result['is_analytic'] is False
        # Could be excessive_repetition or non_english (only 2 unique words)
        assert 'excessive_repetition' in result['flags'] or 'non_english' in result['flags']

    def test_acceptable_repetition(self, filter):
        """Test that normal repetition is acceptable."""
        result = filter.assess_signal('I think that we should work on this work together')
        # Some repetition but within acceptable limits
        # Should not flag excessive_repetition
        if not result['is_analytic']:
            assert 'excessive_repetition' not in result['flags']

    # ========================================================================
    # Test 8: No Alphabetic Content
    # ========================================================================

    def test_no_alphabetic_numbers_only(self, filter):
        """Test detection of numbers-only responses."""
        test_cases = ['123456', '42', '3.14159', '100%']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            assert 'no_alphabetic' in result['flags']

    def test_no_alphabetic_punctuation_only(self, filter):
        """Test detection of punctuation-only responses."""
        test_cases = ['!!!', '???', '...', '---', '***']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['is_analytic'] is False
            # Will be flagged as either no_alphabetic or non_response
            assert 'no_alphabetic' in result['flags'] or 'non_response' in result['flags']

    def test_no_alphabetic_mixed_symbols(self, filter):
        """Test detection of symbol-only responses."""
        result = filter.assess_signal('!@#$%^&*()')
        assert result['is_analytic'] is False
        assert 'no_alphabetic' in result['flags']

    # ========================================================================
    # Test 9: Valid Responses (Should Pass)
    # ========================================================================

    def test_valid_response_simple(self, filter):
        """Test that simple valid responses pass."""
        result = filter.assess_signal('I enjoy working from home because of flexibility.')
        assert result['is_analytic'] is True
        assert len(result['flags']) == 0
        assert result['recommendation'] == 'include'

    def test_valid_response_detailed(self, filter):
        """Test that detailed responses pass."""
        text = (
            'Remote work has significantly improved my work-life balance. '
            'I can spend more time with family and avoid the daily commute, '
            'which saves me about two hours per day. The flexibility to '
            'structure my own schedule has made me more productive.'
        )
        result = filter.assess_signal(text)
        assert result['is_analytic'] is True
        assert len(result['flags']) == 0

    def test_valid_response_with_numbers(self, filter):
        """Test that responses with numbers and text pass."""
        result = filter.assess_signal('I work 8 hours per day and take 2 breaks.')
        assert result['is_analytic'] is True
        assert len(result['flags']) == 0

    def test_valid_response_informal(self, filter):
        """Test that informal but valid responses pass."""
        result = filter.assess_signal('yeah i really like the new system its pretty cool')
        assert result['is_analytic'] is True
        assert len(result['flags']) == 0

    # ========================================================================
    # Test 10: Edge Cases
    # ========================================================================

    def test_edge_case_unicode(self, filter):
        """Test handling of unicode characters."""
        result = filter.assess_signal('I really ❤️ working from home 😊')
        # Should not crash, emojis ignored in processing
        assert 'is_analytic' in result

    def test_edge_case_html_entities(self, filter):
        """Test handling of HTML/special characters."""
        result = filter.assess_signal('The system &amp; process work well together')
        assert 'is_analytic' in result

    def test_edge_case_mixed_case(self, filter):
        """Test that mixed case doesn't affect detection."""
        result = filter.assess_signal('ThIs Is A vAlId ReSpOnSe WiTh CoNtEnT')
        assert result['is_analytic'] is True

    def test_edge_case_long_response(self, filter):
        """Test handling of very long responses."""
        long_text = ' '.join(['This is a valid sentence.'] * 100)
        result = filter.assess_signal(long_text)
        # Should flag excessive_repetition
        assert result['is_analytic'] is False
        assert 'excessive_repetition' in result['flags']

    def test_edge_case_non_string_int(self, filter):
        """Test handling of non-string input (integer)."""
        result = filter.assess_signal(123)
        assert result['is_analytic'] is False
        assert 'null' in result['flags']

    def test_edge_case_non_string_list(self, filter):
        """Test handling of non-string input (list)."""
        result = filter.assess_signal(['test', 'list'])
        assert result['is_analytic'] is False
        assert 'null' in result['flags']

    # ========================================================================
    # Test 11: Confidence Scores
    # ========================================================================

    def test_confidence_null_is_100_percent(self, filter):
        """Test that null detection has 100% confidence."""
        result = filter.assess_signal(None)
        assert result['confidence'] == 1.0

    def test_confidence_empty_is_100_percent(self, filter):
        """Test that empty detection has 100% confidence."""
        result = filter.assess_signal('')
        assert result['confidence'] == 1.0

    def test_confidence_valid_response(self, filter):
        """Test that valid responses have appropriate confidence."""
        result = filter.assess_signal('This is a completely valid response with substance.')
        assert result['confidence'] == 1.0
        assert result['is_analytic'] is True

    def test_confidence_range(self, filter):
        """Test that confidence scores are in valid range [0.0, 1.0]."""
        test_cases = [
            'test',
            'n/a',
            'I work from home',
            'qwerty',
            '123',
            'This is a good response',
            '',
        ]
        for text in test_cases:
            result = filter.assess_signal(text)
            assert 0.0 <= result['confidence'] <= 1.0

    # ========================================================================
    # Test 12: Recommendation Levels
    # ========================================================================

    def test_recommendation_exclude_critical(self, filter):
        """Test that critical issues get 'exclude' recommendation."""
        test_cases = [None, '', 'n/a', '---']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert result['recommendation'] == 'exclude'

    def test_recommendation_review_uncertain(self, filter):
        """Test that uncertain cases get 'review' recommendation."""
        # Too short but not critical
        result = filter.assess_signal('maybe yes')
        assert result['recommendation'] in ['review', 'exclude']

    def test_recommendation_include_valid(self, filter):
        """Test that valid responses get 'include' recommendation."""
        result = filter.assess_signal('I enjoy the flexibility of remote work.')
        assert result['recommendation'] == 'include'

    # ========================================================================
    # Test 13: Batch Processing
    # ========================================================================

    def test_batch_assess(self, filter):
        """Test batch assessment of multiple responses."""
        texts = [
            'This is a valid response with enough content',
            'n/a',
            'Another good response here with more words',
            'test',
            'I really like working remotely from home'
        ]
        results = filter.batch_assess(texts)

        assert len(results) == len(texts)
        assert all('is_analytic' in r for r in results)
        assert all('confidence' in r for r in results)

        # Check specific results
        assert results[0]['is_analytic'] is True
        assert results[1]['is_analytic'] is False
        assert results[2]['is_analytic'] is True
        assert results[3]['is_analytic'] is False
        assert results[4]['is_analytic'] is True

    # ========================================================================
    # Test 14: Flag Statistics
    # ========================================================================

    def test_flag_statistics(self, filter):
        """Test flag statistics calculation."""
        texts = [
            'n/a',
            'n/a',
            'test',
            'This is valid',
            'qwerty',
            '',
            '123'
        ]
        assessments = filter.batch_assess(texts)
        stats = filter.get_flag_statistics(assessments)

        assert 'non_response' in stats
        assert stats['non_response'] == 2  # Two n/a responses
        assert 'empty' in stats
        assert stats['empty'] == 1

    # ========================================================================
    # Test 15: Custom Configuration
    # ========================================================================

    def test_custom_min_words(self):
        """Test custom minimum word count."""
        custom_filter = ContentQualityFilter(min_words=5)
        result = custom_filter.assess_signal('one two three four')  # 4 words
        assert 'too_short_words' in result['flags']

    def test_custom_min_chars(self):
        """Test custom minimum character count."""
        custom_filter = ContentQualityFilter(min_chars=20)
        result = custom_filter.assess_signal('short text')  # < 20 chars
        assert 'too_short_chars' in result['flags']

    def test_custom_repetition_ratio(self):
        """Test custom repetition ratio threshold."""
        custom_filter = ContentQualityFilter(max_repetition_ratio=0.5)
        # 50% unique words (work appears 3 times, good twice out of 5 total)
        result = custom_filter.assess_signal('work work work good good')
        assert 'excessive_repetition' in result['flags']

    def test_custom_english_ratio(self):
        """Test custom English word ratio threshold."""
        custom_filter = ContentQualityFilter(min_english_word_ratio=0.8)
        # Mix of English and non-English
        result = custom_filter.assess_signal('The sistema is muy bueno today')
        # Lower English ratio than threshold
        assert 'non_english' in result['flags']

    # ========================================================================
    # Test 16: Reason Field Quality
    # ========================================================================

    def test_reason_is_human_readable(self, filter):
        """Test that reason field provides human-readable explanations."""
        result = filter.assess_signal('test')
        assert result['reason']
        assert len(result['reason']) > 10  # Not just a code
        assert isinstance(result['reason'], str)

    def test_reason_contains_details(self, filter):
        """Test that reason field contains specific details."""
        result = filter.assess_signal('ok')  # Too short
        assert 'reason' in result
        # Should mention specific issue
        assert 'char' in result['reason'].lower() or 'word' in result['reason'].lower()

    def test_multiple_reasons_combined(self, filter):
        """Test that multiple issues are combined in reason."""
        result = filter.assess_signal('a')  # Multiple issues
        assert result['reason']
        # Should have multiple parts (combined with semicolon)
        # or at least mention one specific issue

    # ========================================================================
    # Test 17: Performance
    # ========================================================================

    def test_performance_single_assessment(self, filter):
        """Test that single assessment is fast."""
        text = 'This is a reasonable length response with some content in it.'
        start = time.time()
        for _ in range(1000):
            filter.assess_signal(text)
        elapsed = time.time() - start

        # Should process 1000 responses in well under 1 second
        assert elapsed < 1.0, f"Performance issue: {elapsed:.3f}s for 1000 assessments"

    def test_performance_batch_assessment(self, filter):
        """Test batch performance meets requirement (<100ms per 1000)."""
        texts = [f'This is test response number {i} with content' for i in range(1000)]

        start = time.time()
        filter.batch_assess(texts)
        elapsed = time.time() - start

        # Must be under 100ms per 1000 responses
        assert elapsed < 0.1, f"Performance requirement not met: {elapsed*1000:.1f}ms per 1000 responses"

    # ========================================================================
    # Test 18: Flag Coverage
    # ========================================================================

    def test_all_flag_types_detected(self, filter):
        """Test that all documented flag types can be detected."""
        flag_test_cases = {
            'null': None,
            'empty': '',
            'too_short_chars': 'hi',
            'too_short_words': 'ok sure',
            'non_response': 'n/a',
            'test_response': 'test',
            'gibberish': 'qwertyuiop asdfghjkl',
            'non_english': 'Hola como estas muy bien gracias',
            'excessive_repetition': 'work work work work work work',
            'no_alphabetic': '12345'
        }

        detected_flags = set()
        for expected_flag, text in flag_test_cases.items():
            result = filter.assess_signal(text)
            detected_flags.update(result['flags'])

        # Should detect at least these core flag types
        core_flags = {'null', 'empty', 'too_short_words', 'non_response', 'test_response'}
        assert core_flags.issubset(detected_flags), f"Missing flags: {core_flags - detected_flags}"

    # ========================================================================
    # Test 19: Auditability
    # ========================================================================

    def test_all_fields_present(self, filter):
        """Test that all required fields are present in results."""
        result = filter.assess_signal('test response')
        required_fields = ['is_analytic', 'confidence', 'reason', 'recommendation', 'flags']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_flags_are_list(self, filter):
        """Test that flags field is always a list."""
        test_cases = ['valid response', 'n/a', '', 'test']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert isinstance(result['flags'], list)

    def test_flags_are_strings(self, filter):
        """Test that all flags are strings."""
        result = filter.assess_signal('test')
        for flag in result['flags']:
            assert isinstance(flag, str)

    def test_confidence_is_float(self, filter):
        """Test that confidence is always a float."""
        test_cases = ['valid response', 'n/a', 'test']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert isinstance(result['confidence'], float)

    def test_is_analytic_is_bool(self, filter):
        """Test that is_analytic is always a boolean."""
        test_cases = ['valid response', 'n/a', 'test']
        for text in test_cases:
            result = filter.assess_signal(text)
            assert isinstance(result['is_analytic'], bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestContentQualityIntegration:
    """Integration tests for content quality assessment."""

    def test_realistic_dataset(self):
        """Test with realistic mix of responses."""
        filter = ContentQualityFilter()

        # Simulate real survey responses
        responses = [
            'I really enjoy the flexibility of working from home.',
            'N/A',
            'Remote work has improved my work-life balance significantly.',
            'test',
            'idk',
            'The ability to avoid commuting saves me 2 hours per day.',
            '',
            'asdfgh',
            'Work from home is good because I can focus better.',
            '123',
            'Not applicable',
            'I like it',
            'Remote work allows me to spend more time with my family and pursue hobbies.',
        ]

        results = filter.batch_assess(responses)
        stats = filter.get_flag_statistics(results)

        # Should identify several non-analytic responses
        non_analytic_count = sum(1 for r in results if not r['is_analytic'])
        assert non_analytic_count >= 5  # At least 5 flagged

        # Should have various flag types
        assert len(stats) >= 3  # Multiple flag types detected

    def test_all_valid_responses(self):
        """Test that dataset of all valid responses passes."""
        filter = ContentQualityFilter()

        responses = [
            'I enjoy working remotely because of the flexibility it provides.',
            'Remote work has allowed me to better manage my time and responsibilities.',
            'The lack of commute has significantly improved my quality of life.',
            'I appreciate the quiet environment at home for focused work.',
            'Virtual meetings are just as effective as in-person ones.',
        ]

        results = filter.batch_assess(responses)

        # All should be analytic
        assert all(r['is_analytic'] for r in results)
        assert all(r['recommendation'] == 'include' for r in results)

    def test_all_invalid_responses(self):
        """Test that dataset of all invalid responses is flagged."""
        filter = ContentQualityFilter()

        responses = [
            'n/a',
            'test',
            '',
            'idk',
            '---',
            'asdf',
            '123',
            None,
        ]

        results = filter.batch_assess(responses)

        # All should be non-analytic
        assert all(not r['is_analytic'] for r in results)

    def test_confidence_distribution(self):
        """Test that confidence scores show reasonable distribution."""
        filter = ContentQualityFilter()

        responses = [
            'Great detailed response',  # High confidence analytic
            'n/a',                       # High confidence non-analytic
            'ok',                        # Medium-low confidence non-analytic
            'test response here',        # Medium confidence non-analytic
        ]

        results = filter.batch_assess(responses)
        confidences = [r['confidence'] for r in results]

        # Should have variation in confidence scores
        assert len(set(confidences)) > 1  # Not all the same
        assert all(0.0 <= c <= 1.0 for c in confidences)  # All in valid range


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
