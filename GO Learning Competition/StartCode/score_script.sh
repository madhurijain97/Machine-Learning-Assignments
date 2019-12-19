python3 ./go_play.py -n 5 -p1 my -p2 random -t 50  | grep Black | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p2 my -p1 random -t 50 | grep White | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p1 my -p2 greedy -t 50  | grep Black | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p2 my -p1 greedy -t 50  | grep White | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p1 my -p2 smart -t 50  | grep Black | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p2 my -p1 smart -t 50  | grep White | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p1 my -p2 aggressive -t 50  | grep Black | cut -d ":" -f 2 | cut -d "%" -f 1
python3 ./go_play.py -n 5 -p2 my -p1 aggressive -t 50  | grep White | cut -d ":" -f 2 | cut -d "%" -f 1=5