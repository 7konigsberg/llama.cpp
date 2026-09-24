#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cstdio>
#include <vector>

#undef NDEBUG
#include <cassert>

// the ngram-hint drafter proposes from token sequences hinted for the request; no model is needed

static llama_tokens draft(common_speculative * spec, const llama_tokens & prompt, llama_token sampled,
                          int32_t n_max = -1) {
    llama_tokens result;

    auto & dp = common_speculative_get_draft_params(spec, 0);
    dp.drafting = true;
    dp.n_max    = n_max;
    dp.pos0     = (llama_pos) prompt.size();
    dp.id_last  = sampled;
    dp.prompt   = &prompt;
    dp.result   = &result;

    common_speculative_draft(spec);
    common_speculative_accept(spec, 0, 0);

    return result;
}

int main() {
    llama_backend_init();

    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_HINT };

    common_speculative_ptr spec(common_speculative_init(params, 1));
    assert(spec);

    const llama_tokens hint = { 100, 101, 102, 103, 104, 105 };

    common_speculative_begin(spec.get(), 0, {});
    common_speculative_hint(spec.get(), 0, hint);

    // a three-token suffix inside the hint drafts what follows it
    assert((draft(spec.get(), { 1, 2, 100, 101 }, 102) == llama_tokens{ 103, 104, 105 }));

    // a two-token suffix does too
    assert((draft(spec.get(), { 7, 103 }, 104) == llama_tokens{ 105 }));

    // the hint's first token on its own enters the hint
    assert((draft(spec.get(), { 1, 2 }, 100) == llama_tokens{ 101, 102, 103, 104, 105 }));

    // a lone token that does not start a hint drafts nothing, even though it is in one
    assert(draft(spec.get(), { 1, 2 }, 103).empty());

    // nothing in common with the hint drafts nothing
    assert(draft(spec.get(), { 1, 2, 3 }, 4).empty());

    // the last token of a hint has nothing after it
    assert(draft(spec.get(), { 103, 104 }, 105).empty());

    // n_max caps the draft
    assert((draft(spec.get(), { 100 }, 101, 2) == llama_tokens{ 102, 103 }));

    // a second hint is searched too; the longest matching key wins
    common_speculative_hint(spec.get(), 0, { 200, 101, 102, 201 });
    assert((draft(spec.get(), { 1, 100, 101 }, 102) == llama_tokens{ 103, 104, 105 }));
    assert((draft(spec.get(), { 1, 200, 101 }, 102) == llama_tokens{ 201 }));

    // begin clears the hints
    common_speculative_begin(spec.get(), 0, {});
    assert(draft(spec.get(), { 100, 101 }, 102).empty());

    // empty hints are ignored
    common_speculative_hint(spec.get(), 0, {});
    assert(draft(spec.get(), { 100, 101 }, 102).empty());

    spec.reset();
    llama_backend_free();

    printf("test-speculative-ngram-hint: ok\n");

    return 0;
}
