# Notes

Things the project owner asked to have noted. Add new entries at the bottom, oldest first, under a date
heading. Write each one in the owner's terms, with a short pointer to related code or docs where it helps.
The learnings log in `PRINCIPLES.md` is separate: it is for findings from experiments.

## 2026-10-10
- Possible downstream interest: mind reading, opponent control, and opponent simulation, i.e. bots that
  inspect, predict by simulating, or manipulate their opponent. The axelrod library tags these as
  `inspects_source` / `manipulates_source` / `manipulates_state` (only Darwin is tagged in v4.14).
  The current contract rules them out: a bot sees only `Obs`. Supporting them would need a contract
  extension, e.g. giving a bot its opponent's `act` and state, and should be designed deliberately
  rather than bolted on.
