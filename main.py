import pandas
from bertopic import BERTopic
from translate import Translator


global chat_history_file, first_messages_output_file, topics_output_file

chat_history_file = 'Data/chats.csv'
first_messages_output_file = 'Data/first_messages.csv'
unique_messages_output_file = 'Data/unique_messages.csv'
grouped_output_file = 'Data/grouped chat history.csv'
def parse_chat_history(file_path):
    try:

        data_frame = pandas.read_csv(file_path,  lineterminator = '\n')
        print(data_frame)
        return data_frame
    except:
        print("не запарсилось")

###
def select_first_messages(df):

    blacklist = ['здравствуйте%', 'добрый день%']

    first_messages = df.groupby('session_id').first()


    for index, row in first_messages.iterrows():#прохожу по файлу получившемуся файлу и склеиваю сообщение
        if len(row['text']) < 10 or (row['text'] in blacklist):
            try:
                next_message = df[(df['session_id'] == index) & (df['role'] == 'user')]['text'].iloc[1]
                first_messages.loc[index, 'text'] = f"{row['text']} {next_message}"
            except:
                print("Вышло за пределы")

    first_messages.to_csv(first_messages_output_file, index=False)

def save_unique_messages(data_frame):

    df = data_frame.drop_duplicates(subset="text")

    df = df["text"].unique()

    df.to_csv(topics_output_file, index=False)

def group_messages_by_topics(df):

    translator = Translator(from_lang="Russian", to_lang="English")
    model = BERTopic()
    topics = model.fit_transform(translator.translate(df['message']))


    df.to_csv(grouped_output_file, index=False)

history_df = parse_chat_history(chat_history_file)
#select_first_messages(history_df)
save_unique_messages(history_df)
group_messages_by_topics(history_df)
