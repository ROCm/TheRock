# Strix Testing Methodology: Black Box vs White Box

## Overview

This document explains the differences between black box and white box testing approaches for Strix platforms (gfx1151/gfx1150).

---

## Black Box Testing

### Definition
Testing the system **without knowledge of internal implementation**.

**Focus:** WHAT the system does  
**Perspective:** User's view  
**Knowledge Required:** Requirements, specifications, expected behavior

### What You Test
- Does the test run on Strix hardware?
- Does it produce correct results?
- Does it pass/fail as expected?
- Does it meet performance requirements?

### What You DON'T Test
- How the test script detects Strix
- Internal conditional logic
- Code paths taken
- Variable assignments

### Example: Test Execution on Strix
```
INPUT: Run hipblaslt test on Strix Halo Windows

EXPECTED OUTPUT: Test completes successfully with quick tests

BLACK BOX VIEW: "It works!" ✅

What we verify:
- Test executable runs
- No crashes
- Correct results produced
- Performance acceptable

What we DON'T verify:
- Internal test script logic
- How environment variables are loaded
- Code branches taken
- Filter construction
```

---

## White Box Testing

### Definition
Testing the system **with knowledge of internal implementation**.

**Focus:** HOW the system works internally  
**Perspective:** Developer's view  
**Knowledge Required:** Source code, architecture, algorithms

### What You Test
- Does the code correctly detect gfx1151?
- Does the Windows + gfx1151 branch execute?
- Is test_type correctly set to "quick"?
- Is the GTest filter properly constructed?

### What You DO Test
- Code path execution
- Conditional logic flow
- Variable state changes
- Branch coverage

### Example: Memory Constraint Logic
```
CODE: if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
          test_type = "quick"

WHITE BOX VIEW:
✅ Line 28: condition evaluated
✅ Both sub-conditions checked
✅ True branch taken
✅ Line 29: test_type assigned "quick"
✅ Variable changed from "full" to "quick"
```

---

## Comparison for Strix

| Aspect | Black Box | White Box |
|--------|-----------|-----------|
| **What is tested** | Functionality, requirements | Code structure, logic |
| **Focus** | Does it work on Strix? | How does Strix detection work? |
| **Test basis** | Requirements, specs | Source code |
| **Knowledge needed** | What Strix should do | How Strix code is implemented |
| **Tester** | QA, End-user | Developer, Code reviewer |
| **Coverage measure** | Requirements coverage | Code coverage (lines, branches) |
| **Defects found** | Functional bugs, wrong results | Logic errors, unhandled cases |
| **Example** | "hipblaslt runs on Strix Windows" | "gfx1151 detection code executes" |

---

## Example 1: Memory Constraint Handling

### Black Box Approach
```
Test Scenario: hipblaslt on Windows Strix Halo

Test Steps:
1. Run hipblaslt test on Windows gfx1151
2. Observe test behavior
3. Check results

Expected Behavior:
- Test completes successfully
- Only quick tests run (not full suite)
- No out-of-memory errors
- Completes in reasonable time

Verdict:
- If quick tests run → PASS ✅
- If full tests run → FAIL ❌ (memory issue likely)
- If crashes → FAIL ❌

Black Box Conclusion:
"The test runs quick tests on Windows Strix and passes"
```

### White Box Approach
```
Test Scenario: Verify memory constraint detection logic

Test Cases:

TC1: Verify condition detection
Code: if AMDGPU_FAMILIES == "gfx1151" and platform == "windows":
Test:
  - Set AMDGPU_FAMILIES = "gfx1151" ✅
  - Set platform = "windows" ✅
  - Condition evaluates to TRUE ✅
  - Coverage: Line 28 executed ✅

TC2: Verify test_type override
Code: test_type = "quick"
Test:
  - Initial test_type = "full" ✅
  - After override: test_type = "quick" ✅
  - Variable changed correctly ✅
  - Coverage: Line 29 executed ✅

TC3: Verify filter construction
Code: if test_type == "quick":
        test_filter.append("--gtest_filter=*quick*")
Test:
  - test_type is "quick" ✅
  - Filter list updated ✅
  - Correct filter string ✅

TC4: Verify negative cases
Test:
  - gfx1151 + Linux → test_type stays "full" ✅
  - gfx1150 + Windows → test_type stays "full" ✅
  - Coverage: False branches tested ✅

White Box Conclusion:
"The code correctly detects Windows gfx1151, overrides test_type,
and constructs the appropriate filter through these specific code paths"
```

---

## Example 2: Test Matrix Generation

### Black Box Approach
```
Input:
- Platform: Linux
- GPU Family: gfx1151
- Request: Generate test matrix

Expected Output:
- JSON with list of tests
- Each test has configuration
- Strix-supported tests included
- Excluded tests not present

Test:
1. Run fetch_test_configurations.py
2. Check output JSON
3. Verify tests present:
   ✅ rocblas in list
   ✅ hipblas in list
4. Verify excluded tests absent:
   ✅ rccl NOT in list (multi-GPU)

Verdict: PASS if output is correct
```

### White Box Approach
```
Test Cases:

TC1: Verify test_matrix iteration
Code: for key in test_matrix:
Test:
  - All 21 tests iterated ✅
  - No tests skipped in loop ✅

TC2: Verify platform filtering
Code: if platform in test_matrix[key]["platform"]:
Test:
  - rocblas: ["linux", "windows"] → Linux match ✅
  - Branch coverage: True/False paths ✅

TC3: Verify exclusion logic
Code: if "exclude_family" in test_matrix[key] and ...
Test:
  - rocsparse has exclude_family ✅
  - Windows in exclusion dict ✅
  - gfx1151 in list ✅
  - All three conditions TRUE → excluded ✅

TC4: Verify shard array generation
Code: [i + 1 for i in range(total_shards)]
Test:
  - total_shards=4 → [1, 2, 3, 4] ✅
  - List comprehension executes ✅

TC5: Verify data flow
  - test_matrix → filtering → output_matrix → JSON ✅
  - Data transformation correct at each step ✅
```

---

## When to Use Each for Strix

### Use Black Box Testing When:

1. **Testing user-facing behavior**
   - "Does this test run on Strix hardware?"
   - "Do I get correct results?"

2. **Validating requirements**
   - "Does Strix support all required libraries?"
   - "Are performance targets met?"

3. **Integration/System testing**
   - "Does the full CI pipeline work for Strix?"
   - "Do tests run end-to-end?"

4. **Acceptance testing**
   - "Can users run tests on Strix?"
   - "Does it meet specifications?"

### Use White Box Testing When:

1. **Testing internal logic**
   - "Does the Strix detection code work correctly?"
   - "Are all branches executed?"

2. **Code review and verification**
   - "Is the memory constraint logic sound?"
   - "Are edge cases handled?"

3. **Debugging issues**
   - "Why does Windows Strix behave differently?"
   - "Which code path causes the error?"

4. **Ensuring code coverage**
   - "Are all Strix-specific branches tested?"
   - "Do we have 90% code coverage?"

---

## Complementary Nature

### Both Approaches Are Needed

**Black Box Testing:**
- ✅ Test runs successfully
- ✅ Results are correct
- ✅ Performance acceptable

**White Box Testing:**
- ✅ Strix detection code executes
- ✅ Memory constraint logic triggers
- ✅ All code paths covered

**Together:** Complete confidence in both **functionality** and **implementation**

---

## Strix Test Strategy Recommendation

### Balanced Approach

**Black Box (60%):**
- Test all 21 libraries on Strix hardware
- Verify results correctness
- Check performance
- System-level testing

**White Box (40%):**
- Test all Strix-specific code paths
- Verify conditional logic
- Ensure code coverage
- Test error handling

---

## Summary

### Black Box for Strix:
- **WHAT:** Tests work on Strix hardware
- **HOW:** Run tests, check results
- **GOAL:** Functional correctness
- **VIEW:** User's perspective

### White Box for Strix:
- **WHAT:** Strix detection logic is correct
- **HOW:** Trace code, test branches
- **GOAL:** Implementation correctness
- **VIEW:** Developer's perspective

### Best Practice:
Use **BOTH** approaches:
- Black box to verify Strix functionality works
- White box to verify Strix code is correct internally

This ensures both **the right thing is built** (black box) and **it's built right** (white box).

