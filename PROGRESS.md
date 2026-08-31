
#### Correction, same night: re-run at the shipped clamp and the gates DO earn their keep

The caveat above was the whole result. Re-running the same sweep with
`max_step_m` at its shipped 0.5 rather than the script's 0.1 default reverses the
conclusion:

    min_margin  consistency   track median   over the bar   worst run
       2.0          1.5          0.38 m           2           4.41 m   <- shipped
       1.5          1.5          0.45 m           3           4.80 m
       1.0          99 (off)     0.44 m           2           5.29 m

**The shipped setting is the best of the three on every column**, and turning the
gates off is worse on the tail (4.41 -> 5.29 m) - which is what a gate against
gross mismatches is for. The finding twenty minutes earlier, that `min_margin`
2.0 was costing accuracy and `consistency_m` did nothing, was an artefact of
sweeping them at a clamp five times tighter than the one that ships: with
corrections capped at 0.1 m the loop is so slow that rejecting half the fixes
dominates everything, and with them capped at 0.5 m it is the rejections that
keep the tail bounded. The two parameters interact and cannot be tuned apart.

Nothing was changed on the strength of the first sweep, which is the only reason
this costs a paragraph instead of a regression. The gates stay at 2.0 and 1.5.
