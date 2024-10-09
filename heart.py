import numpy as np
import cv2
import math
import random


def find_area(figure):
    x_mid = sum([it[0] for it in figure]) // len(figure)
    y_mid = sum([it[1] for it in figure]) // len(figure)
    area = 0
    for k in range(-1, len(figure) - 1):
        point1 = figure[k]
        point2 = figure[k + 1]
        a = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        b = math.sqrt((point2[0] - x_mid) ** 2 + (point2[1] - y_mid) ** 2)
        c = math.sqrt((point1[0] - x_mid) ** 2 + (point1[1] - y_mid) ** 2)
        p = (a + b + c) / 2
        area += math.sqrt(p * (p - a) * (p - b) * (p - c))
    return round(area, 5)


def neighbourhood(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    # Main settings
    picture = 'img/heart/fourcameracut-8.jpg'
    # picture = 'img/vessels/sosudecut-2.jpg'

    # Filter settings
    averaging_parameter = 0.009
    neighbourhood_distance = 5
    point_alienation_parameter = 80
    area_alienation_parameter = 35
    min_points_count_in_one_area = 3

    # Experiment settings
    gauss_floor = 5
    threshold_variants = [24, 30, 36, 42, 48, 54, 60, 66, 72, 78]

    # Areas parameters
    num_of_areas = 4
    max_points_count_in_one_area = 25

    # Result parameters
    result_points_count = 0
    result_gauss_radius = 0
    result_threshold_parameter = 0
    result_points_list = []

    for threshold_parameter in threshold_variants:
        for gauss_radius in range(gauss_floor, 31):
            points = []
            # print(gauss_radius, threshold_parameter)
            # Reading image
            # print(find_area([[1, 4], [3, 5], [4, 4], [4, 2], [2, 2]]))
            font = cv2.FONT_HERSHEY_COMPLEX
            img2 = cv2.imread(picture, cv2.IMREAD_COLOR)
            # img2 = cv2.blur(img2, (25, 25))
            # Reading same image in another
            # variable and converting to gray scale.
            img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE)

            # 2 variants of blur
            img = cv2.blur(img, (gauss_radius, gauss_radius))
            # img = cv2.GaussianBlur(img, (gauss_radius, gauss_radius), 0)

            # Converting image to a binary image
            # ( black and white only image).
            _, threshold = cv2.threshold(img, threshold_parameter, 108, cv2.THRESH_BINARY)

            # Detecting contours in image.
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Going through every contour found in the image.
            for cnt in contours:

                contour = []
                approx = cv2.approxPolyDP(cnt, averaging_parameter * cv2.arcLength(cnt, True), True)

                # Draws boundary of contours.
                cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

                # Used to flatt the array containing the coordinates of the vertices
                n = approx.ravel()
                i = 0
                for _ in n:
                    if i % 2 == 0:
                        x = n[i]
                        y = n[i + 1]
                        contour.append([int(x), int(y)])
                    i += 1
                points.append(contour)

            # Area filter (center of the image)
            area_filter = []
            for i in range(len(points)):
                new_contour = [item for item in points[i] if (300 <= item[0] <= 700) and (250 <= item[1] <= 600)]
                if len(new_contour) > min_points_count_in_one_area:
                    area_filter.append(new_contour)

            # Locality filter (methods "Neighbourhoods" and "The point of alienation")
            locality_filter = []
            for t in range(len(area_filter)):
                bad_indexes = [False for _ in range(len(area_filter[t]))]
                for i in range(len(area_filter[t]) - 1):
                    exclusion = True
                    for j in range(i + 1, len(area_filter[t])):
                        if neighbourhood(area_filter[t][i], area_filter[t][j]) <= neighbourhood_distance:
                            bad_indexes[i] = True
                            bad_indexes[j] = True
                        if neighbourhood(area_filter[t][i], area_filter[t][j]) <= point_alienation_parameter:
                            exclusion = False
                    if exclusion:
                        bad_indexes[i] = True

                new_contour = [area_filter[t][i] for i in range(len(area_filter[t])) if not bad_indexes[i]]
                if len(new_contour) > min_points_count_in_one_area:
                    locality_filter.append(new_contour)

            # Distance filter (delete unnecessary areas)
            current_areas = [True for _ in range(len(locality_filter))]
            for i in range(len(locality_filter)):
                can_delete = True
                for j in range(len(locality_filter)):
                    if i != j and current_areas[i] and current_areas[j]:
                        for point1 in locality_filter[i]:
                            for point2 in locality_filter[j]:
                                if neighbourhood(point1, point2) <= area_alienation_parameter:
                                    can_delete = False
                                    break
                            if not can_delete:
                                break
                    if not can_delete:
                        break
                if can_delete:
                    current_areas[i] = False

            distance_filter = [locality_filter[t] for t in range(len(locality_filter)) if current_areas[t]] \
                if len(locality_filter) > 2 else locality_filter

            # Solving filter (deletes solving if it exceeds limit)
            can_add_areas = True

            # a) Max points filter (exceeding of point limit)
            current_points_count = sum([len(elem) for elem in distance_filter])
            for elem in distance_filter:
                if len(elem) > max_points_count_in_one_area:
                    can_add_areas = False
                    break

            # b) Comon area filter (exceeding of heart's size limit)
            north = 10 ** 6
            south = 0
            west = 10 ** 6
            east = 0
            for contour in distance_filter:
                for point in contour:
                    north = min(north, point[1])
                    south = max(south, point[1])
                    west = min(west, point[0])
                    east = max(east, point[0])

            if abs(north - south) > 250 or abs(west - east) > 250:
                can_add_areas = False

            # c) Final checking
            if len(distance_filter) == num_of_areas and current_points_count >= result_points_count and can_add_areas:
                result_points_count = current_points_count
                result_gauss_radius = gauss_radius
                result_threshold_parameter = threshold_parameter
                result_points_list = distance_filter

    if result_points_count != 0:
        distance_filter = result_points_list

        # Output information
        print(f'Optimal gauss radius is {result_gauss_radius} px')
        print(f'Optimal threshold parameter is {result_threshold_parameter} pt')
        print(f'Points coordinates: {distance_filter}')
        img2 = cv2.imread(picture, cv2.IMREAD_COLOR)
        for t in range(len(distance_filter)):
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            print(f'Area {t + 1}: {len(distance_filter[t])} points')
            for item in distance_filter[t]:
                img2 = cv2.circle(img2, (item[0], item[1]), radius=3, color=random_color, thickness=-1)
            # cv2.putText(img2, str(item[0]) + " " + str(item[1]), (item[0], item[1]), font, 0.3, random_color)
            for i in range(-1, len(distance_filter[t]) - 1):
                cv2.line(img2, (distance_filter[t][i][0], distance_filter[t][i][1]),
                         (distance_filter[t][i + 1][0], distance_filter[t][i + 1][1]), random_color, thickness=1)

        print(f'Total: {sum([len(elem) for elem in distance_filter])} points')
        cv2.imshow('image2', img2)

        # print(results)
        # Exiting the window if 'q' is pressed on the keyboard.
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
