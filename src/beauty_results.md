# Beauty Results

```
$ poetry run python src/main.py 
Loading data...
First pass: Counting interactions...

Before filtering:
Total users: 22363
Total items: 12101

After filtering (>= 5 interactions):
Valid users: 22363
Valid items: 12101

Second pass: Loading filtered data...

Final filtered reviews: 198502

=== Chronological Analysis ===

Found 19424 users with temporal ordering issues:

Issue 1:
User: A1YJEY40YUW4SE
Original sequence:
  2014-01-29: 7806397051
  2012-02-01: B0020YLEYK
  2014-01-29: B002WLWX82
  2011-10-17: B004756YJA
  2011-10-17: B004ZT0SSG
Sorted sequence:
  2011-10-17: B004756YJA
  2011-10-17: B004ZT0SSG
  2012-02-01: B0020YLEYK
  2014-01-29: 7806397051
  2014-01-29: B002WLWX82

Issue 2:
User: A60XNB876KYML
Original sequence:
  2014-04-17: 7806397051
  2014-04-17: B0000YUX4O
  2014-02-14: B0009P4PZC
  2014-04-16: B00812ZWOS
  2014-03-30: B009HULFLW
  2014-03-30: B00BZ1QN2C
  2014-04-06: B00G2TQNZ4
Sorted sequence:
  2014-02-14: B0009P4PZC
  2014-03-30: B009HULFLW
  2014-03-30: B00BZ1QN2C
  2014-04-06: B00G2TQNZ4
  2014-04-16: B00812ZWOS
  2014-04-17: 7806397051
  2014-04-17: B0000YUX4O

Issue 3:
User: A3G6XNM240RMWA
Original sequence:
  2013-09-05: 7806397051
  2014-02-27: B00011JI88
  2013-11-18: B001MP471K
  2013-09-08: B002S8TOYU
  2014-03-15: B003ATNYJC
  2013-09-05: B003H8180I
  2014-05-16: B003ZS6ONQ
  2013-09-05: B00538TSMU
  2014-03-02: B00C1F13CQ
Sorted sequence:
  2013-09-05: 7806397051
  2013-09-05: B003H8180I
  2013-09-05: B00538TSMU
  2013-09-08: B002S8TOYU
  2013-11-18: B001MP471K
  2014-02-27: B00011JI88
  2014-03-02: B00C1F13CQ
  2014-03-15: B003ATNYJC
  2014-05-16: B003ZS6ONQ

Issue 4:
User: A1PQFP6SAJ6D80
Original sequence:
  2013-12-07: 7806397051
  2013-10-04: B00027D8IC
  2013-10-04: B002PMLGOU
  2013-09-13: B0030HKJ8I
  2013-12-07: B004Z40048
  2013-10-04: B00BN1MPPS
Sorted sequence:
  2013-09-13: B0030HKJ8I
  2013-10-04: B00027D8IC
  2013-10-04: B002PMLGOU
  2013-10-04: B00BN1MPPS
  2013-12-07: 7806397051
  2013-12-07: B004Z40048

Issue 5:
User: A38FVHZTNQ271F
Original sequence:
  2013-10-18: 7806397051
  2013-11-20: B002BGDLDO
  2013-11-20: B003VWZCMK
  2013-11-19: B007EHWDTS
  2013-11-04: B008LQX8J0
  2013-11-19: B009DDGHFC
  2013-11-03: B009PZVOF6
  2014-01-05: B00DAYGJVW
  2013-11-20: B00DQ2ILQY
Sorted sequence:
  2013-10-18: 7806397051
  2013-11-03: B009PZVOF6
  2013-11-04: B008LQX8J0
  2013-11-19: B007EHWDTS
  2013-11-19: B009DDGHFC
  2013-11-20: B002BGDLDO
  2013-11-20: B003VWZCMK
  2013-11-20: B00DQ2ILQY
  2014-01-05: B00DAYGJVW

Temporal Statistics:
Date range: 2002-06-11 to 2014-07-22

Users with multiple reviews on same day: 36802

Sample cases of multiple reviews per day:
User A1YJEY40YUW4SE: 2 reviews on 2014-01-29
User A1YJEY40YUW4SE: 2 reviews on 2011-10-17
User A60XNB876KYML: 2 reviews on 2014-04-17
User A60XNB876KYML: 2 reviews on 2014-03-30
User A3G6XNM240RMWA: 3 reviews on 2013-09-05
First pass: Counting interactions...

Before filtering:
Total users: 22363
Total items: 12101

After filtering (>= 5 interactions):
Valid users: 22363
Valid items: 12101

Second pass: Loading filtered data...

Final filtered reviews: 198502

Loading filtered metadata...
Loaded metadata for 12101 items with 0 errors

=== Sample User Histories ===

User: A02849582PREZYNEI31CV (5 reviews)
  2014-03-30: B007RT19V6, rating=5.0
    Summary: great tools !...
  2014-03-30: B007TL60IE, rating=5.0
    Summary: very cute!!!...
  2014-03-30: B0083QNBCM, rating=5.0
    Summary: cute cheap brushes...
  2014-03-30: B0087O4XKE, rating=1.0
    Summary: comes off easy!!!...
  2014-03-30: B00AZBSPQA, rating=2.0
    Summary: pop off easy!...

User: A1AN74S2DBATOJ (6 reviews)
  2014-02-24: B00A19WNC8, rating=5.0
    Summary: These are one of my Favorite designs!!...
  2014-03-09: B001169LHK, rating=4.0
    Summary: Nail Glue......
  2014-03-09: B0084BPHY6, rating=4.0
    Summary: Buff away!...
  2014-03-22: B005JD4NZQ, rating=4.0
    Summary: FUN!!...
  2014-03-22: B006U95N34, rating=4.0
    Summary: They are doing just what they're suppose to do!...
  2014-03-28: B007USPWS0, rating=4.0
    Summary: Cute!...

User: A2Q9EK9WKGFGCG (8 reviews)
  2011-05-09: B0043TURKC, rating=4.0
    Summary: So silky!...
  2013-10-06: B00D6EDGYE, rating=4.0
    Summary: Lightweight coverage and great scent!...
  2014-01-20: B00G2TQNZ4, rating=5.0
    Summary: I love this stuff!...
  2014-04-21: B00IBS9QC6, rating=5.0
    Summary: Loved it, great stuff!...
  2014-05-16: B00CGN9LQ8, rating=5.0
    Summary: I love this brush...
  2014-05-20: B00JJVG6HC, rating=5.0
    Summary: Great for dry itchy scalp!...
  2014-05-20: B00JL2TURM, rating=5.0
    Summary: Great for acne prone skin...
  2014-07-15: B00EYSNWXG, rating=5.0
    Summary: It's a good curler!...

=== Duplicate Analysis ===
Total reviews: 198502
Exact duplicates: 0
Users with multiple reviews for same item: 0

=== Rating Analysis ===
Overall rating distribution:
  1.0: 10526 (5.3%)
  2.0: 11456 (5.8%)
  3.0: 22248 (11.2%)
  4.0: 39741 (20.0%)
  5.0: 114531 (57.7%)

Users with single rating value:
  User A00700212KB3K0MVESPIY: all 5.0s (9 reviews)
  User A0508779FEO1DUNOSQNX: all 5.0s (5 reviews)
  User A05306962T0DL4FS2RA7L: all 5.0s (5 reviews)
  User A100VQNP6I54HS: all 5.0s (8 reviews)
  User A1010QRG4BH51B: all 5.0s (11 reviews)

Generating embeddings for 12101 items...
Including fields: ['brand', 'category', 'description', 'price', 'sales_rank', 'title']
Processing items 0-5/12101
Embedding dimension: 768
Processing items 1000-1005/12101
Processing items 2000-2005/12101
Processing items 3000-3005/12101
Processing items 4000-4005/12101
Processing items 5000-5005/12101
Processing items 6000-6005/12101
Processing items 7000-7005/12101
Processing items 8000-8005/12101
Processing items 9000-9005/12101
Processing items 10000-10005/12101
Processing items 11000-11005/12101
Processing items 12000-12005/12101

Computing semantic relationships...

Validation data preparation:
Total users processed: 22363
Users skipped (too short): 0
Final validation sequences: 22363

Evaluation data preparation:
Total users processed: 22363
Users skipped (too short): 0
Users with timestamp issues: 19586
Final test sequences: 22363

=== Evaluation Split Verification ===

Found 0 sequences where test item isn't truly last

History length distribution:
  Length 4: 7162 sequences
  Length 5: 4221 sequences
  Length 6: 2680 sequences
  Length 7: 1811 sequences
  Length 8: 1366 sequences
  Length 9: 881 sequences
  Length 10: 695 sequences
  Length 11: 558 sequences
  Length 12: 439 sequences
  Length 13: 361 sequences
  Length 14: 262 sequences
  Length 15: 207 sequences
  Length 16: 213 sequences
  Length 17: 141 sequences
  Length 18: 118 sequences
  Length 19: 112 sequences
  Length 20: 117 sequences
  Length 21: 84 sequences
  Length 22: 66 sequences
  Length 23: 73 sequences
  Length 24: 74 sequences
  Length 25: 61 sequences
  Length 26: 54 sequences
  Length 27: 44 sequences
  Length 28: 44 sequences
  Length 29: 38 sequences
  Length 30: 30 sequences
  Length 31: 38 sequences
  Length 32: 20 sequences
  Length 33: 19 sequences
  Length 34: 21 sequences
  Length 35: 19 sequences
  Length 36: 23 sequences
  Length 37: 24 sequences
  Length 38: 22 sequences
  Length 39: 15 sequences
  Length 40: 17 sequences
  Length 41: 7 sequences
  Length 42: 10 sequences
  Length 43: 10 sequences
  Length 44: 15 sequences
  Length 45: 7 sequences
  Length 46: 11 sequences
  Length 47: 8 sequences
  Length 48: 4 sequences
  Length 49: 8 sequences
  Length 50: 9 sequences
  Length 51: 8 sequences
  Length 52: 9 sequences
  Length 53: 6 sequences
  Length 54: 8 sequences
  Length 55: 2 sequences
  Length 56: 6 sequences
  Length 57: 4 sequences
  Length 58: 7 sequences
  Length 59: 3 sequences
  Length 60: 1 sequences
  Length 62: 4 sequences
  Length 63: 2 sequences
  Length 64: 2 sequences
  Length 65: 4 sequences
  Length 66: 1 sequences
  Length 67: 3 sequences
  Length 68: 4 sequences
  Length 69: 2 sequences
  Length 70: 2 sequences
  Length 71: 2 sequences
  Length 72: 4 sequences
  Length 73: 3 sequences
  Length 74: 3 sequences
  Length 75: 1 sequences
  Length 76: 5 sequences
  Length 77: 2 sequences
  Length 78: 3 sequences
  Length 79: 1 sequences
  Length 80: 1 sequences
  Length 81: 2 sequences
  Length 82: 4 sequences
  Length 83: 1 sequences
  Length 84: 3 sequences
  Length 86: 3 sequences
  Length 92: 2 sequences
  Length 93: 1 sequences
  Length 95: 1 sequences
  Length 96: 1 sequences
  Length 98: 2 sequences
  Length 99: 4 sequences
  Length 106: 1 sequences
  Length 107: 1 sequences
  Length 109: 1 sequences
  Length 114: 2 sequences
  Length 115: 1 sequences
  Length 116: 1 sequences
  Length 118: 1 sequences
  Length 122: 1 sequences
  Length 130: 1 sequences
  Length 148: 2 sequences
  Length 149: 1 sequences
  Length 153: 1 sequences
  Length 181: 1 sequences
  Length 191: 1 sequences
  Length 203: 1 sequences

Processing user-item interactions...
Found 20484 users and 12101 items
Built interaction matrix with 137119 non-zero entries

=== Negative Sampling Debug ===

User: A1TIRS7AIPSVXQ
History length: 5
Total valid items: 12101
Excluded items: 6
Available negatives: 12095

User: A2XO9VWQ7DNFP7
History length: 5
Total valid items: 12101
Excluded items: 6
Available negatives: 12095

User: ALX6TKL1GRPMV
History length: 5
Total valid items: 12101
Excluded items: 6
Available negatives: 12095

User: A3VI56NFNK3K96
History length: 4
Total valid items: 12101
Excluded items: 5
Available negatives: 12096

User: A1YP5WLIHGG136
History length: 8
Total valid items: 12101
Excluded items: 9
Available negatives: 12092

=== Collaborative Matrix Analysis ===
Shape: (12101, 12101)
Density: 0.0107
Mean nonzero value: 0.0743
Max value: 8.4437

Value distribution (nonzero):
  25th percentile: 0.0154
  Median: 0.0385
  75th percentile: 0.0909

=== Running Evaluation ===

=== Starting Evaluation ===
Evaluating:   0%|                                                                                                                        | 0/22363 [00:00<?, ?it/s]
Intermediate stats at sequence 0:
Average candidates per user: 12096
Current metrics:
hit@5: 0.0000
hit@10: 0.0000
ndcg@5: 0.0000
ndcg@10: 0.0000
Evaluating:   4%|████▉                                                                                                         | 999/22363 [00:32<11:12, 31.77it/s]
Intermediate stats at sequence 1000:
Average candidates per user: 12094
Current metrics:
hit@5: 0.0619
hit@10: 0.0839
ndcg@5: 0.0416
ndcg@10: 0.0487
Evaluating:   9%|█████████▋                                                                                                   | 1997/22363 [01:03<10:27, 32.43it/s]
Intermediate stats at sequence 2000:
Average candidates per user: 12096
Current metrics:
hit@5: 0.0650
hit@10: 0.0895
ndcg@5: 0.0421
ndcg@10: 0.0500
Evaluating:  13%|██████████████▌                                                                                              | 3000/22363 [01:36<10:05, 31.97it/s]
Intermediate stats at sequence 3000:
Average candidates per user: 12093
Current metrics:
hit@5: 0.0656
hit@10: 0.0940
ndcg@5: 0.0435
ndcg@10: 0.0527
Evaluating:  18%|███████████████████▍                                                                                         | 3998/22363 [02:07<09:27, 32.38it/s]
Intermediate stats at sequence 4000:
Average candidates per user: 12066
Current metrics:
hit@5: 0.0662
hit@10: 0.0955
ndcg@5: 0.0441
ndcg@10: 0.0536
Evaluating:  22%|████████████████████████▎                                                                                    | 4997/22363 [02:39<08:55, 32.46it/s]
Intermediate stats at sequence 5000:
Average candidates per user: 12096
Current metrics:
hit@5: 0.0626
hit@10: 0.0908
ndcg@5: 0.0417
ndcg@10: 0.0508
Evaluating:  27%|█████████████████████████████▏                                                                               | 6000/22363 [03:10<08:58, 30.40it/s]
Intermediate stats at sequence 6000:
Average candidates per user: 12094
Current metrics:
hit@5: 0.0635
hit@10: 0.0912
ndcg@5: 0.0421
ndcg@10: 0.0510
Evaluating:  31%|██████████████████████████████████                                                                           | 6997/22363 [03:42<09:07, 28.08it/s]
Intermediate stats at sequence 7000:
Average candidates per user: 12096
Current metrics:
hit@5: 0.0657
hit@10: 0.0930
ndcg@5: 0.0442
ndcg@10: 0.0530
Evaluating:  36%|██████████████████████████████████████▉                                                                      | 8000/22363 [04:14<07:46, 30.78it/s]
Intermediate stats at sequence 8000:
Average candidates per user: 12091
Current metrics:
hit@5: 0.0627
hit@10: 0.0907
ndcg@5: 0.0420
ndcg@10: 0.0510
Evaluating:  40%|███████████████████████████████████████████▊                                                                 | 8997/22363 [04:45<06:52, 32.38it/s]
Intermediate stats at sequence 9000:
Average candidates per user: 12084
Current metrics:
hit@5: 0.0632
hit@10: 0.0923
ndcg@5: 0.0422
ndcg@10: 0.0517
Evaluating:  45%|████████████████████████████████████████████████▎                                                           | 10000/22363 [05:17<06:25, 32.10it/s]
Intermediate stats at sequence 10000:
Average candidates per user: 12091
Current metrics:
hit@5: 0.0633
hit@10: 0.0924
ndcg@5: 0.0422
ndcg@10: 0.0516
Evaluating:  49%|█████████████████████████████████████████████████████                                                       | 10998/22363 [05:50<07:49, 24.21it/s]
Intermediate stats at sequence 11000:
Average candidates per user: 12076
Current metrics:
hit@5: 0.0634
hit@10: 0.0934
ndcg@5: 0.0427
ndcg@10: 0.0524
Evaluating:  54%|█████████████████████████████████████████████████████████▉                                                  | 12000/22363 [06:36<05:43, 30.16it/s]
Intermediate stats at sequence 12000:
Average candidates per user: 12052
Current metrics:
hit@5: 0.0624
hit@10: 0.0914
ndcg@5: 0.0421
ndcg@10: 0.0514
Evaluating:  58%|██████████████████████████████████████████████████████████████▊                                             | 12997/22363 [07:09<04:54, 31.77it/s]
Intermediate stats at sequence 13000:
Average candidates per user: 12089
Current metrics:
hit@5: 0.0628
hit@10: 0.0918
ndcg@5: 0.0428
ndcg@10: 0.0521
Evaluating:  63%|███████████████████████████████████████████████████████████████████▌                                        | 13998/22363 [07:56<05:05, 27.39it/s]
Intermediate stats at sequence 14000:
Average candidates per user: 12097
Current metrics:
hit@5: 0.0625
hit@10: 0.0916
ndcg@5: 0.0427
ndcg@10: 0.0520
Evaluating:  67%|████████████████████████████████████████████████████████████████████████▍                                   | 14998/22363 [08:28<03:50, 32.00it/s]
Intermediate stats at sequence 15000:
Average candidates per user: 12097
Current metrics:
hit@5: 0.0629
hit@10: 0.0919
ndcg@5: 0.0427
ndcg@10: 0.0520
Evaluating:  72%|█████████████████████████████████████████████████████████████████████████████▎                              | 16000/22363 [09:01<03:18, 32.13it/s]
Intermediate stats at sequence 16000:
Average candidates per user: 12093
Current metrics:
hit@5: 0.0630
hit@10: 0.0924
ndcg@5: 0.0426
ndcg@10: 0.0520
Evaluating:  76%|██████████████████████████████████████████████████████████████████████████████████                          | 17000/22363 [09:33<02:47, 32.04it/s]
Intermediate stats at sequence 17000:
Average candidates per user: 12085
Current metrics:
hit@5: 0.0639
hit@10: 0.0929
ndcg@5: 0.0433
ndcg@10: 0.0527
Evaluating:  80%|██████████████████████████████████████████████████████████████████████████████████████▉                     | 17999/22363 [10:06<03:12, 22.67it/s]
Intermediate stats at sequence 18000:
Average candidates per user: 12079
Current metrics:
hit@5: 0.0642
hit@10: 0.0930
ndcg@5: 0.0436
ndcg@10: 0.0528
Evaluating:  85%|███████████████████████████████████████████████████████████████████████████████████████████▋                | 18997/22363 [10:39<01:45, 31.92it/s]
Intermediate stats at sequence 19000:
Average candidates per user: 12091
Current metrics:
hit@5: 0.0637
hit@10: 0.0928
ndcg@5: 0.0432
ndcg@10: 0.0526
Evaluating:  89%|████████████████████████████████████████████████████████████████████████████████████████████████▌           | 19999/22363 [11:11<01:21, 28.98it/s]
Intermediate stats at sequence 20000:
Average candidates per user: 12096
Current metrics:
hit@5: 0.0633
hit@10: 0.0925
ndcg@5: 0.0429
ndcg@10: 0.0523
Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍      | 21000/22363 [11:44<00:42, 32.14it/s]
Intermediate stats at sequence 21000:
Average candidates per user: 12091
Current metrics:
hit@5: 0.0631
hit@10: 0.0921
ndcg@5: 0.0426
ndcg@10: 0.0520
Evaluating:  98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▏ | 21999/22363 [12:16<00:11, 32.10it/s]
Intermediate stats at sequence 22000:
Average candidates per user: 12095
Current metrics:
hit@5: 0.0632
hit@10: 0.0923
ndcg@5: 0.0428
ndcg@10: 0.0521
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22363/22363 [12:28<00:00, 29.88it/s]

Successfully evaluated 22363/22363 sequences

Final Results:

Results for Beauty dataset:
------------------------------
Metric          Score     
------------------------------
hit@10          0.0923
hit@5           0.0632
ndcg@10         0.0521
ndcg@5          0.0428
------------------------------
```