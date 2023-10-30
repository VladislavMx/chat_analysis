import pandas as pd
from bertopic import BERTopic


# Парсинг CSV файла
def parse_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def select_first_user_messages(df):
    user_messages = []

    chat_groups = df.groupby("session_id")
    for group_key, group_df in chat_groups:
        first_user_message = group_df.loc[group_df["role"] == "user", "text"].values[0]
        if len(first_user_message) < 10 or first_user_message.lower().startswith("здравствуйте"):
            second_user_message = group_df.loc[group_df["role"] == "user", "text"].values[1]
            first_user_message += " " + second_user_message

        user_messages.append(first_user_message)

    return user_messages



def save_unique_messages(messages, output_file):
    unique_messages = list(set(messages))
    pd.DataFrame(unique_messages, columns=["messages"]).to_csv(output_file, index=None)



def group_messages_by_topics(messages, output_file):
    topic_model = BERTopic(language="Multilanguage")
    topics, _ = topic_model.fit_transform(messages)
    pd.DataFrame({"messages": messages, "topics": topics}).to_csv(output_file, index=None)


def main():
    file_path = "chats.csv"
    output_file_unique_messages = "unique_messages.csv"
    output_file_topics = "message_topics.csv"

    df = parse_csv_file(file_path)
    user_messages = select_first_user_messages(df)
    save_unique_messages(user_messages, output_file_unique_messages)
    group_messages_by_topics(user_messages, output_file_topics)


if __name__ == "__main__":
    main()

