import praw
import xlwt
from xlwt import Workbook


reddit = praw.Reddit(user_agent='Comment Extraction (by /u/)',
                     client_id='', client_secret="")

users = []

subreddit = reddit.subreddit('The_Donald')
subreddit.quaran.opt_in()

for post in subreddit.top('week', limit=5):
    users.append(post.author)

subreddit = reddit.subreddit('aww')
for post in subreddit.top('week', limit=5):
    users.append(post.author)

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
count = 0

for user in users:
    user = reddit.redditor(str(user))
    for comment in user.comments.controversial(limit=50):
        sheet1.write(count, 0, str(comment.body))
        sheet1.write(count, 1, str(user))
        count += 1

wb.save('reddit_users.xls')
